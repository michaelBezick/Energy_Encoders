import time

import matplotlib.pyplot as plt
import torch
import polytensor

from annealing_classes import RNN, RNN_Concat, RNN_Tensorized, Variational_Free_Energy
from Energy_Encoder_Classes import BVAE

device = "cuda"
epochs = 1000
lr = 5e-4
batch_size = 100
warmup_steps = 0
temperature = 1
N_gradient_descent = 1
N_samples = 50
log_step_size = 10
min_energy_repeat_threshold = 600
vector_length = 64
num_vector_samples = 30

print_vector = False
plot = True
save_vectors = True
##################################################
num_vars = 64

num_per_degree = [num_vars]
sample_fn = lambda: torch.randn(1, device="cuda")
terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

terms = polytensor.generators.denseFromSparse(terms, num_vars)
terms.append(torch.randn(num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

second_degree_model = BVAE(energy_fn, torch.randn(1), h_dim=128)
second_degree_model.load_state_dict(torch.load("./Models/QUBO_order_2/epoch=4999-step=100000.ckpt")['state_dict'])

terms.append(torch.randn(num_vars, num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

third_degree_model = BVAE(energy_fn, torch.randn(1), h_dim=128)
third_degree_model.load_state_dict(torch.load("./Models/QUBO_order_3/epoch=4999-step=100000.ckpt")['state_dict'])

terms.append(torch.randn(num_vars, num_vars, num_vars, num_vars))
energy_fn = polytensor.DensePolynomial(terms)

fourth_degree_model = BVAE(energy_fn, torch.randn(1), h_dim=128)
fourth_degree_model.load_state_dict(torch.load("./Models/QUBO_order_4/epoch=4999-step=100000.ckpt")['state_dict'])


model_list = [second_degree_model, third_degree_model, fourth_degree_model]

for experiment_number, model in enumerate(model_list):
    model = model.to(device)
    energy_loss = Variational_Free_Energy(
        model.energy_fn, N_samples=N_samples, batch_size=batch_size
    )
    energy_loss = energy_loss.to(device)
    rnn = RNN_Tensorized().to(device)

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

        if min_energy_repeats > min_energy_repeat_threshold:
            break

        if epoch % log_step_size == 0:
            print(f"Epoch: {epoch}")
            print(f"Min Energy: {min_energy}")
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
    time_per_epoch = elapsed_time / (epoch + warmup_steps)

    save_dir = ""
    if experiment_number == 0:
        save_dir = "./Models/QUBO_order_2/"
    elif experiment_number == 1:
        save_dir = "./Models/QUBO_order_3/"
    else:
        save_dir = "./Models/QUBO_order_4/"

    print(f"Elapsed time: {elapsed_time}")
    print(f"Time per epoch: {time_per_epoch}")
    sigma = torch.tensor(unique_vector_list)
    print(sigma.size())
    if save_vectors:
        torch.save(sigma, save_dir + "neural_annealing_vectors.pt")

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

        plt.savefig(save_dir + "Average_Energy_Plot.png")
