import numpy as np
from numpy.lib.function_base import average
import polytensor
import torch
from torch.utils.data import DataLoader

from annealing_classes import RNN, Variational_Free_Energy
from Energy_Encoder_Classes import BVAE, CorrelationalLoss
from Energy_Encoder_Classes import LabeledDataset, LabeledDatasetForVectors
import torch.nn.functional as F
import matplotlib.pyplot as plt


def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))


num_retraining_iterations = 100
energy_function_retraining_epochs = 1
device = "cuda"
epochs = 100  # in the graphs 100 seems to be a safe place for convergence
lr = 5e-4
batch_size = 100
warmup_steps = 0
temperature = 1
N_gradient_descent = 10
N_samples = 50
log_step_size = 10
min_energy_repeat_threshold = 100000
vector_length = 64
num_vector_samples = 30

print_vector = False
plot = False
save_vectors = False
RNN_type = "Simple_RNN"
##################################################
energy_loss_fn = CorrelationalLoss(1, 1, 1)

num_vars = 64

total_time_list = []
time_per_vector_list = []
list_of_unique_vector_lists = []


num_per_degree = [num_vars]
sample_fn = lambda: torch.randn(1, device="cuda")
terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

terms = polytensor.generators.denseFromSparse(terms, num_vars)
terms.append(torch.randn(num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

second_degree_model = BVAE.load_from_checkpoint(
    "./Models/QUBO_order_2/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)
terms.append(torch.randn(num_vars, num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

third_degree_model = BVAE.load_from_checkpoint(
    "./Models/QUBO_order_3/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)

terms.append(torch.randn(num_vars, num_vars, num_vars, num_vars))
energy_fn = polytensor.DensePolynomial(terms)

fourth_degree_model = BVAE.load_from_checkpoint(
    "./Models/QUBO_order_4/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)

dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

dataset = clamp_output(dataset, 0.5)

labels = torch.load("../Files/FOM_labels_new.pt")

labeled_dataset = LabeledDataset(dataset, labels)
print(f"Original dataset length: {labeled_dataset.__len__()}")

train_loader = DataLoader(
    labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

model_list = [second_degree_model, third_degree_model, fourth_degree_model]


for experiment_number, model in enumerate(model_list):
    if experiment_number != 2:
        continue

    model = model.to(device)
    model.scale = model.scale.to(device)
    model.energy_fn.coefficients = model.energy_fn.coefficients.to(device)
    model.sum_of_squares_begin = model.sum_of_squares_begin.to(device)
    energy_loss = Variational_Free_Energy(
        model.energy_fn, N_samples=N_samples, batch_size=batch_size
    )

    energy_loss = energy_loss.to(device)
    rnn = RNN().to(device)

    rnn_optimizer = torch.optim.Adam(params=rnn.parameters(), lr=lr)

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

    delay_temp = 0

    for retraining_iteration in range(num_retraining_iterations):
        """IDEA: RETRAIN ENERGY FUNCTION FIRST, THEN PERFORM ANNEALING."""

        new_surrogate_model = model.energy_fn
        energy_fn_lr = 1e-3
        surrogate_model_optimizer = torch.optim.Adam(params=new_surrogate_model.parameters(), lr=energy_fn_lr)
        correlational_loss = CorrelationalLoss(10.0, 0.01, 0.0)
        sampler = torch.multinomial
        latent_vector_dim = 64
        num_logits = 2
        norm_weight = 10
        energy_weight = 1e-2

        average_energies_list = []
        total_loss_list = []
        norm_loss_list = []
        correlational_loss_list = []
        pearson_correlation_list = []
        surrogate_model_retraining_batch_size = 500

        """NEED TO CREATE NEW LABELED DATASET FOR VECTORS"""
        new_vector_dataset_labeled = LabeledDatasetForVectors()

        for batch in train_loader:
                with torch.no_grad():
                    images, labels = batch
                    images = images.cuda().float()
                    labels = labels.cuda().float()
                    #need to encode to vector
                    logits = model.vae.encode(images)
                    probabilities = F.softmax(logits, dim=2)
                    probabilities_condensed = probabilities.view(-1, num_logits)
                    sampled_vector = sampler(probabilities_condensed, 1, True).view(
                        batch_size, latent_vector_dim
                    )
                    valid_vector = model.scale_vector_copy_gradient(sampled_vector, probabilities)
                    new_vector_dataset_labeled.vectors.extend(valid_vector)
                    new_vector_dataset_labeled.FOMs.extend(labels)



        vector_loader = DataLoader(
            new_vector_dataset_labeled, batch_size=surrogate_model_retraining_batch_size, shuffle=True, drop_last=True
        )

        print(f"New dataset length: {new_vector_dataset_labeled.__len__()}")

        """ENERGY FUNCTION RETRAINING"""
        for energy_function_epoch in range(energy_function_retraining_epochs):
            for i, batch in enumerate(train_loader):
                images, labels = batch
                images = images.cuda().float()
                labels = labels.cuda().float()
                #need to encode to vector
                logits = model.vae.encode(images)
                """Actually I want a new dataset that only has vectors. I don't need to encode each time"""
                probabilities = F.softmax(logits, dim=2)
                probabilities_condensed = probabilities.view(-1, num_logits)
                sampled_vector = sampler(probabilities_condensed, 1, True).view(
                    batch_size, latent_vector_dim
                )
                valid_vector = model.scale_vector_copy_gradient(sampled_vector, probabilities)
                original_sampled_vector_with_gradient = valid_vector
                sampled_energies = model.energy_fn(original_sampled_vector_with_gradient)
                sampled_energies = torch.squeeze(sampled_energies)
                cl_loss = correlational_loss(labels, sampled_energies) * energy_weight #ORDER MATTERS FOR INFORMATION TRACKING
                norm = model.calc_norm(model.energy_fn.coefficients)
                norm_loss = F.mse_loss(norm, torch.ones_like(norm)) * norm_weight
                total_loss = cl_loss + norm_loss

                surrogate_model_optimizer.zero_grad()
                total_loss.backward()
                surrogate_model_optimizer.step()

                if i % 10 == 0:
                    average_energies_list.append(torch.mean(sampled_energies).detach().cpu().numpy())
                    total_loss_list.append(total_loss.detach().cpu().numpy())
                    norm_loss_list.append(norm_loss.detach().cpu().numpy())
                    correlational_loss_list.append(cl_loss.detach().cpu().numpy())
                    pearson_correlation_list.append(correlational_loss.correlation.detach().cpu().numpy())

                    print(f"------------------------------")
                    print(f"Epoch: {energy_function_epoch}\tIteration: {i}")
                    print(f"Loss: {total_loss}")
                    print(f"Correlational Loss: {cl_loss}")
                    print(f"Norm Loss: {norm_loss}")
                    print(f"Norm: {model.calc_norm(model.energy_fn.coefficients)}")
                    correlational_loss.print_info()

        new_energies = []
        new_labels = []
        for i, batch in enumerate(train_loader):
            with torch.no_grad():
                images, labels = batch
                images = images.cuda().float()
                labels = labels.cuda().float()
                new_labels.extend(labels.cpu().numpy())
                logits = model.vae.encode(images)
                probabilities = F.softmax(logits, dim=2)
                probabilities_condensed = probabilities.view(-1, num_logits)
                sampled_vector = sampler(probabilities_condensed, 1, True).view(
                    batch_size, latent_vector_dim
                )
                valid_vector = model.scale_vector_copy_gradient(sampled_vector, probabilities)
                original_sampled_vector_with_gradient = valid_vector
                sampled_energies = model.energy_fn(original_sampled_vector_with_gradient)
                sampled_energies = torch.squeeze(sampled_energies)
                new_energies.extend(sampled_energies.cpu().numpy())

        t = list(range(1, len(average_energies_list) + 1))

        plt.figure()
        plt.plot(t, average_energies_list)
        plt.savefig("Average_Energies.png")
        plt.close()

        plt.figure()
        plt.plot(t, total_loss_list)
        plt.savefig("Total_Losses.png")
        plt.close()

        plt.figure()
        plt.plot(t, norm_loss_list)
        plt.savefig("Norm_Losses.png")
        plt.close()

        plt.figure()
        plt.plot(t, correlational_loss_list)
        plt.savefig("Correlational_Losses.png")
        plt.close()

        plt.figure()
        plt.plot(t, pearson_correlation_list)
        plt.savefig("Pearson_Correlations.png")
        plt.close()

        plt.figure()
        plt.scatter(new_labels, new_energies)
        plt.savefig("New_Scatter_Plot.png")
        plt.close()
        
        exit()

        """AFTER ANNEALING IS PERFORMED, I NEEEEEEED TO SORT THE DISCOVERED VECTORS BY EFFICIENCY.
            THEN, TAKE N NUMBER OF VECTORS FROM EACH INTERVAL [0, 0.1), ... [0.9, 1.0].
            PREFERABLY THE N VECTORS FROM THE UPPER END OF EACH INTERVAL.
        """



        model.energy_fn = new_surrogate_model

        """PERFORM ANNEALING"""

        min_energy = 0
        for epoch in range(epochs):
            if epoch > delay_temp:
                temperature -= 1 / (epochs - delay_temp)
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

            if torch.min(model.energy_fn(sigma)) < min_energy:
                min_energy = torch.min(model.energy_fn(sigma))

            if epoch % log_step_size == 0:
                print(f"Retraining Iteration: {retraining_iteration}\tEpoch: {epoch}")
                print(f"Min Energy: {min_energy}")
                print(f"Average Energy: {torch.mean(model.energy_fn(sigma))}")
                # average_energies.append(torch.mean(model.energy_fn(sigma)).item())
                # temperatures.append(temperature)

            rnn_optimizer.zero_grad()
            loss.backward()
            rnn_optimizer.step()

            if epoch % N_gradient_descent == 0:
                sigma = torch.bernoulli(sigma_hat)

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # time_per_vector = elapsed_time / len(unique_vector_list)
        #
        # total_time_list.append(elapsed_time)
        # time_per_vector_list.append(time_per_vector)
        #
        # time_per_epoch = elapsed_time / (epoch + warmup_steps)

        """NOW NEED TO PUT ALL NEW VECTORS INTO DATASET TO HAVE ENERGY FUNCTION LEARN THEIR CORRELATION"""
        """Need to calcuate the energy and FOM of each vector and update dataset."""

    save_dir = ""
    if experiment_number == 0:
        save_dir = "./Models/QUBO_order_2/"
    elif experiment_number == 1:
        save_dir = "./Models/QUBO_order_3/"
    elif experiment_number == 2:
        save_dir = "./Models/QUBO_order_4/"
    else:
        save_dir = "./Models/Composite/"

    list_of_unique_vector_lists.append(len(unique_vector_list))
    sigma = torch.tensor(unique_vector_list)
    print(sigma.size())

    if save_vectors:
        torch.save(sigma, save_dir + RNN_type + "neural_annealing_vectors.pt")

    # if plot:
    #     total_steps = epoch + warmup_steps
    #     steps = list(range(0, total_steps, log_step_size))
    #
    #     plt.figure()
    #
    #     plt.plot(
    #         steps,
    #         average_energies,
    #         label="Average Variational Classical Annealing Solution Energy",
    #         marker="o",
    #         linestyle="-",
    #     )
    #     plt.xlabel("Transition Steps")
    #     plt.ylabel("Average Energy")
    #     plt.title("Average Energy of Solutions versus Transition Steps")
    #
    #     plt.legend()
    #
    #     plt.savefig(save_dir + RNN_type + "Average_Energy_Plot.png")
