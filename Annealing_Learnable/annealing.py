import time

import matplotlib.pyplot as plt
import polytensor
import torch

from annealing_classes import (
    RNN,
    Variational_Free_Energy,
    Variational_Free_Energy_modified_for_specific_value,
)
from Energy_Encoder_Classes import BVAE, CorrelationalLoss

device = "cuda"
epochs = 1000
lr = 5e-4
batch_size = 100
warmup_steps = 0
temperature = 1
N_gradient_descent = 1
N_samples = 50
log_step_size = 10
min_energy_repeat_threshold = 100000
vector_length = 64
num_vector_samples = 30

print_vector = False
plot = True
save_vectors = True
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

composite_model = BVAE.load_from_checkpoint(
    "./Models/Composite/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)

model_list = [second_degree_model, third_degree_model, fourth_degree_model]
min_energy_suggestion_list = [-5, -20, -475]

for experiment_number, model in enumerate(model_list):
    # if experiment_number != 3:
    #     continue
    model = model.to(device)
    energy_loss = Variational_Free_Energy(
        model.energy_fn, N_samples=N_samples, batch_size=batch_size
    )

    """ADDED"""
    energy_loss = Variational_Free_Energy_modified_for_specific_value(
        model.energy_fn, N_samples=N_samples, batch_size=batch_size
    )
    """"""

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

    start_time = time.time()

    unique_vector_list = []
    unique_vector_set = set()

    # sigma is sampled
    # sigma_hat is probabililties
    for i in range(warmup_steps):
        sigma_hat = rnn(sigma)

        loss = energy_loss(sigma_hat, temperature)
        if i % log_step_size == 0:
            print(f"Average Energy: {torch.mean(model.energy_fn(sigma))}")
            print(f"Loss: {loss}")
            print(f"i: {i}")
            average_energies.append(torch.mean(model.energy_fn(sigma)).item())
            temperatures.append(temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # move to next state after n gradient descent steps
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

        if torch.min(model.energy_fn(sigma)) < min_energy_suggestion_list[experiment_number]:
            print(f"Suggestion: {min_energy_suggestion_list[experiment_number]}")
            break

        if min_energy_repeats > min_energy_repeat_threshold:
            break

        # if len(unique_vector_list) > 100:
        #     break

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

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_vector = elapsed_time / len(unique_vector_list)

    total_time_list.append(elapsed_time)
    time_per_vector_list.append(time_per_vector)

    time_per_epoch = elapsed_time / (epoch + warmup_steps)

    save_dir = ""
    if experiment_number == 0:
        save_dir = "./Models/QUBO_order_2/"
    elif experiment_number == 1:
        save_dir = "./Models/QUBO_order_3/"
    elif experiment_number == 2:
        save_dir = "./Models/QUBO_order_4/"
    else:
        save_dir = "./Models/Composite/"

    print(f"Elapsed time: {elapsed_time}")
    print(f"Time per epoch: {time_per_epoch}")
    list_of_unique_vector_lists.append(len(unique_vector_list))
    sigma = torch.tensor(unique_vector_list)
    print(sigma.size())
    if save_vectors:
        torch.save(sigma, save_dir + RNN_type + "neural_annealing_vectors.pt")

    if plot:
        total_steps = epoch + warmup_steps
        steps = list(range(0, total_steps, log_step_size))

        plt.figure()

        plt.plot(
            steps,
            average_energies,
            label="Average Variational Classical Annealing Solution Energy",
            marker="o",
            linestyle="-",
        )
        plt.xlabel("Transition Steps")
        plt.ylabel("Average Energy")
        plt.title("Average Energy of Solutions versus Transition Steps")

        plt.legend()

        plt.savefig(save_dir + RNN_type + "Average_Energy_Plot.png")

for i in range(3):
    print(f"Experiment {i}:, total time: {total_time_list[i]}, time per vector: {time_per_vector_list[i]}, num_vectors: {list_of_unique_vector_lists[i]}")
