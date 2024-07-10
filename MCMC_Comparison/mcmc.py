import math as m
import time

import matplotlib.pyplot as plt
import numpy as np
import polytensor
import torch
import torch.nn.functional as F
from Energy_Encoder_Classes import BVAE, CorrelationalLoss

scale = torch.Tensor([0.0, 1.0]).cuda()
device = "cuda"


def MCMC_step(initial_vectors, energy_fn, temperature):
    batch_size = initial_vectors.size()[0]
    num_elements = initial_vectors.size()[1]

    indices = torch.randint(0, num_elements, (batch_size, 1), device="cuda")

    # get new spins corresponding to indices
    new_spins = torch.randint(0, 2, (batch_size, 1), device="cuda")

    new_spins = F.one_hot(new_spins, 2).float()
    new_spins = torch.einsum("ijk,k->ij", new_spins, scale)

    transitioned_vectors = initial_vectors.scatter(dim=1, index=indices, src=new_spins)

    # # need to move to transitioned vectors with acceptance probabilities
    initial_energy = torch.squeeze(energy_fn(initial_vectors))
    transitioned_energy = torch.squeeze(energy_fn(transitioned_vectors))
    e_matrix = m.e * torch.ones((batch_size), device=device)
    acceptance_prob_RHS = torch.pow(
        e_matrix, (initial_energy - transitioned_energy) / (temperature)
    )
    acceptance_prob_LHS = torch.ones((batch_size), device=device)
    acceptance_probability = torch.min(acceptance_prob_LHS, acceptance_prob_RHS)

    acceptance_sample = torch.bernoulli(acceptance_probability).unsqueeze(1).int()
    acceptance_sample_expanded = acceptance_sample.expand(batch_size, num_elements)

    # if acceptance_sample = 1, move to next one
    output = torch.where(
        acceptance_sample_expanded == 1, transitioned_vectors, initial_vectors
    )

    return output


energy_loss_fn = CorrelationalLoss(1, 1, 1)

num_vars = 64

num_per_degree = [num_vars]
sample_fn = lambda: torch.randn(1, device="cuda")
terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

terms = polytensor.generators.denseFromSparse(terms, num_vars)
terms.append(torch.randn(num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

second_degree_model = BVAE.load_from_checkpoint(
    "../Annealing_Learnable/Models/QUBO_order_2/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)
terms.append(torch.randn(num_vars, num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

third_degree_model = BVAE.load_from_checkpoint(
    "../Annealing_Learnable/Models/QUBO_order_3/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)

terms.append(torch.randn(num_vars, num_vars, num_vars, num_vars))
energy_fn = polytensor.DensePolynomial(terms)

fourth_degree_model = BVAE.load_from_checkpoint(
    "../Annealing_Learnable/Models/QUBO_order_4/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)

batch_size = 100
vector_length = 64
warmup_steps = 0
epochs = 2000
log_step_size = 20
save_vectors = True
plot = True
epsilon = 1e-10

model_list = [second_degree_model, third_degree_model, fourth_degree_model]
for experiment_number, model in enumerate(model_list):

    with torch.no_grad():
        temperature = 1
        model = model.to(device)

        initial_vector = torch.bernoulli(torch.ones(batch_size, vector_length) * 0.5).to(
            device
        )

        average_energies = []
        temperatures = []

        min_energy = 0
        best_vector = torch.zeros(vector_length).to(device)

        start_time = time.time()

        unique_vector_list = []
        unique_vector_set = set()

        min_energy_repeats = 0
        epoch = 0

        vectors = initial_vector

        for epoch in range(epochs):
            temperature -= 1 / (epochs)

            if temperature <= 0:
                temperature = epsilon

            vectors = MCMC_step(vectors, model.energy_fn, temperature)

            # adding vectors to unique vector set
            list_of_vectors = vectors.tolist()
            for vector in list_of_vectors:
                vector_tuple = tuple(vector)
                if vector_tuple not in unique_vector_set:
                    unique_vector_set.add(vector_tuple)
                    unique_vector_list.append(vector)

            # finding min energy, with consideration of repeats
            min_energy_repeats += 1

            try:
                if torch.min(model.energy_fn(vectors)) < min_energy:
                    min_energy_repeats = 0
                    min_energy = torch.min(model.energy_fn(vectors))
                    index = torch.argmin(model.energy_fn(vectors))
                    best_vector = vectors[index, :]
            except:
                print('WEROIWJRWEAKJSD;FLAKJ')
                print(vectors)

            # if min_energy_repeats > min_energy_repeat_threshold:
            #     break

            if epoch % log_step_size == 0:
                print(f"Epoch: {epoch}")
                print(f"Min Energy: {min_energy}")
                average_energies.append(torch.mean(model.energy_fn(vectors)).item())
                temperatures.append(temperature)


        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_epoch = elapsed_time / (epoch + warmup_steps)

        save_dir = ""
        if experiment_number == 0:
            save_dir = "./Models/2nd_Order/"
        elif experiment_number == 1:
            save_dir = "./Models/3rd_Order/"
        else:
            save_dir = "./Models/4th_Order/"

        print(f"Elapsed time: {elapsed_time}")
        print(f"Time per epoch: {time_per_epoch}")
        sigma = torch.tensor(unique_vector_list)
        print(sigma.size())

        if save_vectors:
            torch.save(sigma, save_dir + "simulated_annealing_vectors.pt")

        if plot:
            total_steps = epoch + warmup_steps
            steps = list(range(0, total_steps, log_step_size))

            plt.figure()

            plt.plot(
                steps,
                average_energies,
                label="Average Simulated Annealing Solution Energy",
                marker="o",
                linestyle="-",
            )
            plt.xlabel("Transition Steps")
            plt.ylabel("Average Energy")
            plt.title("Average Energy of Solutions versus Transition Steps")

            plt.legend()

            plt.savefig(save_dir + "Average_Energy_Plot.png")
