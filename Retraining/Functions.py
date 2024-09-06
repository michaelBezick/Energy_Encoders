import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def expand_output(tensor: torch.Tensor):
    x = torch.zeros([tensor.size()[0], 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x


def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))


def create_initial_dataset_for_vectors(
    train_loader, dataset, model, batch_size=100, latent_vector_dim=64, num_logits=2
):
    sampler = torch.multinomial
    for batch in train_loader:
        with torch.no_grad():
            images, labels = batch
            images = images.cuda().float()
            labels = labels.cuda().float()
            logits = model.vae.encode(images)
            probabilities = F.softmax(logits, dim=2)
            probabilities_condensed = probabilities.view(-1, num_logits)
            sampled_vector = sampler(probabilities_condensed, 1, True).view(
                batch_size, latent_vector_dim
            )
            valid_vector = model.scale_vector_copy_gradient(
                sampled_vector, probabilities
            )
            mask = (labels.squeeze() <= 1.0) & (labels.squeeze() >= 0.0)
            filtered_vectors = valid_vector[mask]
            filtered_FOM = labels[mask]
            # dataset.vectors.extend(valid_vector)
            # dataset.FOMs.extend(labels)
            dataset.vectors.extend(filtered_vectors)
            dataset.FOMs.extend(filtered_FOM)

    return dataset


def retrain_surrogate_model(
    train_loader,
    retraining_epochs,
    model,
    correlational_loss_fn,
    lr,
    correlational_loss_weight,
    norm_weight,
):
    surrogate_model_optimizer = torch.optim.Adam(
        params=model.energy_fn.parameters(), lr=lr
    )
    norm = 0

    min_energy = 100
    for epoch in range(retraining_epochs):

        for batch in train_loader:
            vectors, FOMs = batch

            energies = model.energy_fn(vectors)

            if epoch == (retraining_epochs - 1):
                if torch.min(energies) < min_energy:
                    min_energy = torch.min(energies).item()

            cl_loss = (
                correlational_loss_fn(FOMs, energies) * correlational_loss_weight
            )  # ORDER MATTERS FOR INFORMATION TRACKING

            norm = model.calc_norm(model.energy_fn.coefficients)
            norm_loss = F.mse_loss(norm, torch.ones_like(norm)) * norm_weight
            total_loss = cl_loss + norm_loss

            surrogate_model_optimizer.zero_grad()
            total_loss.backward()
            surrogate_model_optimizer.step()

    print(f"Final correlation trained: {correlational_loss_fn.correlation}")
    print(f"Final norm: {norm}")
    print(f"MIN ENERGY: {min_energy}")
    return model, min_energy


def perform_annealing(
    batch_size,
    vector_length,
    device,
    annealing_epochs,
    initial_temperature,
    delay_temp,
    energy_loss,
    rnn,
    num_vector_samples,
    model,
    lr,
    N_gradient_descent,
    lowest_epochs=False,
    epoch_bound=100,
    min_energy_surrogate=100.0,
    energy_mismatch_threshold=1.0,
):
    """NEW IDEA: MAKE IT SHORT CIRCUIT WHEN CLOSE TO MIN ENERGY"""

    min_energy = 100
    unique_vector_set = set()
    unique_vector_list = []
    sigma = torch.bernoulli(torch.ones(batch_size, vector_length) * 0.5).to(device)
    temperature = initial_temperature
    rnn_optimizer = torch.optim.Adam(params=rnn.parameters(), lr=lr)

    for epoch in range(annealing_epochs):
        if epoch > delay_temp:
            temperature -= 1 / (annealing_epochs - delay_temp)
            if temperature <= 0:
                temperature = 0

        sigma_hat = rnn(sigma)

        loss = energy_loss(sigma_hat, temperature)

        # adding vectors to unique vector set

        if (lowest_epochs == False) or (epoch > epoch_bound):
            for i in range(num_vector_samples):
                list_of_vectors = torch.bernoulli(sigma).tolist()
                for vector in list_of_vectors:
                    vector_tuple = tuple(vector)
                    if vector_tuple not in unique_vector_set:
                        unique_vector_set.add(vector_tuple)
                        unique_vector_list.append(vector)

        if torch.min(model.energy_fn(sigma)) < min_energy:
            min_energy = torch.min(model.energy_fn(sigma))

        """GET ENERGY MISMATCH"""
        energy_mismatch_value = min_energy - min_energy_surrogate
        if energy_mismatch_value <= energy_mismatch_threshold:
            print(
                f"Short circuit. Epoch: {epoch}. Energy mismatch: {energy_mismatch_value}"
            )
            break

        # if epoch % log_step_size == 0:

        # print("------------------------------")
        # print(f"Retraining Iteration: {retraining_iteration}\tEpoch: {epoch}")
        # print(f"Min Energy: {min_energy}")
        # print(f"Average Energy: {torch.mean(model.energy_fn(sigma))}")
        # average_energies.append(torch.mean(model.energy_fn(sigma)).item())
        # temperatures.append(temperature)

        rnn_optimizer.zero_grad()
        loss.backward()
        rnn_optimizer.step()

        if epoch % N_gradient_descent == 0:
            sigma = torch.bernoulli(sigma_hat)

    print(f"Min energy reached: {min_energy}")
    return unique_vector_list, min_energy


def calc_efficiencies_of_new_vectors(
    unique_vector_list, device, decoding_batch_size, model, FOM_calculator
):
    unique_vector_tensor = torch.tensor(unique_vector_list, device=device)

    vector_loader_2 = DataLoader(
        dataset=unique_vector_tensor,
        batch_size=decoding_batch_size,
        shuffle=False,
        drop_last=False,
    )

    new_vectors_FOM_list = []
    new_designs = []
    for vector in vector_loader_2:

        decoded_images = model.vae.decode(vector)

        output_expanded = expand_output(decoded_images)

        output_expanded = clamp_output(output_expanded, 0.5)

        FOMs = FOM_calculator(
            torch.permute(output_expanded.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy()
        )

        new_vectors_FOM_list.extend(FOMs.numpy().flatten().tolist())
        new_designs_list = [t.squeeze(0) for t in output_expanded]
        new_designs.extend(new_designs_list)

    return new_vectors_FOM_list, new_designs


def add_new_vectors_to_dataset(
    unique_vector_list,
    new_vectors_FOM_list,
    new_vector_dataset_labeled,
    device,
    threshold=False,
    threshold_value=0.9,
    bound=False,
    lower_bound=0.0,
    upper_bound=1.1,
):

    new_vectors_tensor = torch.tensor(unique_vector_list, device=device)
    new_FOMs_tensor = torch.tensor(new_vectors_FOM_list, device=device)

    if threshold:
        new_combined_first = torch.cat(
            [new_vectors_tensor, new_FOMs_tensor.unsqueeze(1)], dim=1
        )
        efficiency_values = new_combined_first[:, 64]
        mask = efficiency_values >= threshold_value
        filtered = new_combined_first[mask]
        new_combined_first = filtered
        new_vectors_tensor = new_combined_first[:, :64]
        new_FOMs_tensor = new_combined_first[:, 64]

    if bound:

        new_combined_first = torch.cat(
            [new_vectors_tensor, new_FOMs_tensor.unsqueeze(1)], dim=1
        )
        efficiency_values = new_combined_first[:, 64]
        mask = (lower_bound <= efficiency_values) & (efficiency_values <= upper_bound)
        filtered = new_combined_first[mask]
        new_combined_first = filtered
        new_vectors_tensor = new_combined_first[:, :64]
        new_FOMs_tensor = new_combined_first[:, 64]

    new_vector_dataset_labeled.FOMs.extend(new_FOMs_tensor)
    new_vector_dataset_labeled.vectors.extend(new_vectors_tensor)

    temp_FOMs = torch.stack(new_vector_dataset_labeled.FOMs).cpu()
    temp_vectors = torch.stack(new_vector_dataset_labeled.vectors).cpu()
    temp_FOMs = temp_FOMs.unsqueeze(1)
    combined = torch.cat([temp_vectors, temp_FOMs], dim=1)
    new_combined = torch.unique(combined, dim=0)

    new_vectors = new_combined[:, :64]
    new_FOMs = new_combined[:, 64]
    new_vector_dataset_labeled.vectors = []
    new_vector_dataset_labeled.FOMs = []
    new_vector_dataset_labeled.vectors.extend(new_vectors.to(device))
    new_vector_dataset_labeled.FOMs.extend(new_FOMs.to(device))

    return new_vector_dataset_labeled
