import polytensor
import torch

from annealing_classes import (
    RNN,
    Variational_Free_Energy,
)
from Energy_Encoder_Classes import BVAE, CorrelationalLoss

def Annealing_Learnable_and_Matching(model_filename=""):
    device = "cuda"
    epochs = 200
    lr = 5e-4
    batch_size = 100
    warmup_steps = 0
    temperature = 1
    N_gradient_descent = 1
    N_samples = 50
    log_step_size = 10
    vector_length = 64
    num_vector_samples = 30

    print_vector = False
    ##################################################
    energy_loss_fn = CorrelationalLoss(1, 1, 1)
    num_vars = 64

    num_per_degree = [num_vars]
    sample_fn = lambda: torch.zeros(1, device="cuda")
    terms = polytensor.generators.coeffPUBORandomSampler(
        n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
    )

    terms = polytensor.generators.denseFromSparse(terms, num_vars)
    terms.append(torch.zeros(num_vars, num_vars))
    terms.append(torch.zeros(num_vars, num_vars, num_vars))

    energy_fn = polytensor.DensePolynomial(terms)

    # learnable_third_degree_model = BVAE.load_from_checkpoint(
    #     "./Annealing_Learnable/Models/QUBO_order_3/epoch=9999-step=200000.ckpt",
    #     energy_fn=energy_fn,
    #     energy_loss_fn=energy_loss_fn,
    #     h_dim=128,
    # )

    matching_third_degree_model = BVAE.load_from_checkpoint(
        "./Annealing_Matching/Models/QUBO_order_3/epoch=9999-step=200000.ckpt",
        energy_fn=energy_fn,
        energy_loss_fn=energy_loss_fn,
        h_dim=128,
    )

    # model_list = [third_degree_model_learnable, third_degree_model_matching]
    model_list = [matching_third_degree_model]

    # energies_learnable = []
    energies_matching = []

    for experiment_number, model in enumerate(model_list):
        model = model.to(device)
        energy_loss = Variational_Free_Energy(
            model.energy_fn, N_samples=N_samples, batch_size=batch_size
        )
        
        energy_loss = energy_loss.to(device)
        rnn = RNN().to(device)

        optimizer = torch.optim.Adam(params=rnn.parameters(), lr=lr)

        initial_vector = torch.bernoulli(torch.ones(batch_size, vector_length) * 0.5).to(
            device
        )
        sigma = initial_vector

        average_energies = []
        temperatures = []

        min_energy = 0
        best_vector = torch.zeros(vector_length).to(device)


        unique_vector_list = []
        unique_vector_set = set()

        for i in range(warmup_steps):
            sigma_hat = rnn(sigma)

            loss = energy_loss(sigma_hat, temperature)
            if i % log_step_size == 0:
                average_energies.append(torch.mean(model.energy_fn(sigma)).item())
                temperatures.append(temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % N_gradient_descent == 0:
                sigma = torch.bernoulli(sigma_hat)

        min_energy_repeats = 0
        epoch = 0
        for epoch in range(epochs):
            if epoch > 100:
                temperature -= 1 / (epochs - 100)
                if temperature <= 0:
                    temperature = 0
            sigma_hat = rnn(sigma)

            loss = energy_loss(sigma_hat, temperature)

            # adding vectors to unique vector set
            for i in range(num_vector_samples):
                list_of_vectors = torch.bernoulli(sigma).tolist()
                for vector in list_of_vectors:
                    vector_tuple = tuple(vector)
                    if vector_tuple not in unique_vector_set:
                        unique_vector_set.add(vector_tuple)
                        unique_vector_list.append(vector)

            # finding min energy, with consideration of repeats
            min_energy_repeats += 1

            if torch.min(model.energy_fn(sigma)) < min_energy:
                min_energy_repeats = 0
                min_energy = torch.min(model.energy_fn(sigma))
                index = torch.argmin(model.energy_fn(sigma))
                best_vector = sigma[index, :]

            if epoch % log_step_size == 0:
                print(f"Epoch: {epoch}")
                print(f"Min Energy: {min_energy}")
                print(f"Average Energy: {torch.mean(model.energy_fn(sigma))}")
                print(f"Loss: {loss}")
                if print_vector:
                    print(f"{best_vector}")
                average_energies.append(torch.mean(model.energy_fn(sigma)).item())
                temperatures.append(temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % N_gradient_descent == 0:
                sigma = torch.bernoulli(sigma_hat)

        # if experiment_number == 0:
        #     energies_learnable = average_energies
        # else:
        energies_matching = average_energies

        break


    '''CHANGED'''
    return [energies_matching]
