import math as m
import torch.nn.functional as F

import matplotlib.pyplot as plt
import polytensor
import torch

from annealing_classes import (
    RNN,
    RNN_Batch_Norm,
    RNN_Sampler,
    Tensorized_RNN_Sampler,
    RNN_Tensorized,
    Variational_Free_Energy,
    Variational_Free_Energy_New,
)
from Energy_Encoder_Classes import BVAE, CorrelationalLoss

"""Self referential doesn't work"""

device = "cuda"
epochs = 200
lr = 5e-4
batch_size = 100
warmup_steps = 0
temperature = 1
N_gradient_descent = 1
N_samples = 50
log_step_size = 10
min_energy_repeat_threshold = 1000000
vector_length = 64
num_vector_samples = 30
model_type = ""

print_vector = False
plot = True
save_vectors = False
RNN_type = "Simple_RNN_batch_norm"
scale = torch.Tensor([0.0, 1.0]).cuda()
##################################################
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

model_list = [second_degree_model, third_degree_model, fourth_degree_model]

rnn_sampler = RNN_Sampler().cuda()

tensorized_rnn_sampler = Tensorized_RNN_Sampler()

for experiment_number, model in enumerate(model_list):

    model = model.to(device)

    if experiment_number == 0:
        model_type = "Second Degree PUBO"
    elif experiment_number == 1:
        model_type = "Third Degree PUBO"
    elif experiment_number == 2:
        model_type = "Fourth Degree PUBO"
    else:
        print("EJA;SLFKAJDFLKERRROROROROROR")

    rnn_tensorized = RNN_Tensorized().to(device)

    optimizer = torch.optim.Adam(params=rnn_tensorized.parameters(), lr=lr)

    initial_vector = torch.bernoulli(torch.ones(batch_size, vector_length) * 0.5).to(
        device
    )

    sigma = initial_vector

    average_energies_rnn_tensorized = []
    temperatures_rnn_tensorized = []

    min_energy = 0
    best_vector = torch.zeros(vector_length).to(device)

    unique_vector_list = []
    unique_vector_set = set()
    # energy_loss = Variational_Free_Energy(
    #     model.energy_fn, N_samples=N_samples, batch_size=batch_size
    # )
    """CHANGED TO NEW"""
    energy_loss = Variational_Free_Energy_New(model.energy_fn, N_samples=N_samples, batch_size=batch_size)

    """RNN_Tensorized"""

    temperature = 1

    for i in range(warmup_steps):
        sigma_hat = rnn_tensorized(sigma)

        loss = energy_loss(sigma_hat, temperature)
        if i % log_step_size == 0:
            print(f"Average Energy: {torch.mean(model.energy_fn(sigma))}")
            print(f"Loss: {loss}")
            print(f"i: {i}")
            average_energies_rnn_tensorized.append(torch.mean(model.energy_fn(sigma)).item())
            temperatures_rnn_tensorized.append(temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # move to next state after n gradient descent steps
        if i % N_gradient_descent == 0 and i > 0:
            sigma = tensorized_rnn_sampler(rnn_tensorized)
            # sigma = torch.bernoulli(sigma_hat)

    min_energy_repeats = 0
    epoch = 0
    for epoch in range(epochs):
        temperature -= 1 / (epochs)
        if temperature <= 0:
            temperature = 0
        sigma_hat = rnn_tensorized(sigma)

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
            average_energies_rnn_tensorized.append(torch.mean(model.energy_fn(sigma)).item())
            temperatures_rnn_tensorized.append(temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % N_gradient_descent == 0 and epoch > 0:
            sigma = tensorized_rnn_sampler(rnn_tensorized)
            # sigma = torch.bernoulli(sigma_hat)


    save_dir = ""
    if experiment_number == 0:
        save_dir = "./Models/QUBO_order_2/"
    elif experiment_number == 1:
        save_dir = "./Models/QUBO_order_3/"
    else:
        save_dir = "./Models/QUBO_order_4/"

    sigma = torch.tensor(unique_vector_list)

    model = model.to(device)
    energy_loss = Variational_Free_Energy(
        model.energy_fn, N_samples=N_samples, batch_size=batch_size
    )
    energy_loss = energy_loss.to(device)
    rnn = RNN().to(device)
    rnn_bn = RNN_Batch_Norm().to(device)

    optimizer = torch.optim.Adam(params=rnn.parameters(), lr=lr)

    initial_vector = torch.bernoulli(torch.ones(batch_size, vector_length) * 0.5).to(
        device
    )
    sigma = initial_vector

    average_energies_rnn = []
    temperatures_rnn = []

    min_energy = 0
    best_vector = torch.zeros(vector_length).to(device)

    unique_vector_list = []
    unique_vector_set = set()

    """RNN"""

    temperature = 1

    for i in range(warmup_steps):
        sigma_hat = rnn(sigma)

        loss = energy_loss(sigma_hat, temperature)
        if i % log_step_size == 0:
            print(f"Average Energy: {torch.mean(model.energy_fn(sigma))}")
            print(f"Loss: {loss}")
            print(f"i: {i}")
            average_energies_rnn.append(torch.mean(model.energy_fn(sigma)).item())
            temperatures_rnn.append(temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # move to next state after n gradient descent steps
        if i % N_gradient_descent == 0 and i > 0:

            sigma = rnn_sampler(rnn)

            # sigma = torch.bernoulli(sigma_hat)

    min_energy_repeats = 0
    epoch = 0
    for epoch in range(epochs):
        temperature -= 1 / (epochs)
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
            average_energies_rnn.append(torch.mean(model.energy_fn(sigma)).item())
            temperatures_rnn.append(temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % N_gradient_descent == 0 and epoch > 0:

            sigma = rnn_sampler(rnn)
            # sigma = torch.bernoulli(sigma_hat)


    save_dir = ""
    if experiment_number == 0:
        save_dir = "./Models/QUBO_order_2/"
    elif experiment_number == 1:
        save_dir = "./Models/QUBO_order_3/"
    else:
        save_dir = "./Models/QUBO_order_4/"

    sigma = torch.tensor(unique_vector_list)


    """MCMC"""

    epsilon = 1e-10

    with torch.no_grad():
        temperature = 1
        model = model.to(device)

        initial_vector = torch.bernoulli(torch.ones(batch_size, vector_length) * 0.5).to(
            device
        )

        average_energies_MCMC = []
        temperatures_MCMC = []

        min_energy = 0
        best_vector = torch.zeros(vector_length).to(device)


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
                average_energies_MCMC.append(torch.mean(model.energy_fn(vectors)).item())
                temperatures_MCMC.append(temperature)



        save_dir = ""
        if experiment_number == 0:
            save_dir = "./Models/2nd_Order/"
        elif experiment_number == 1:
            save_dir = "./Models/3rd_Order/"
        else:
            save_dir = "./Models/4th_Order/"

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
            average_energies_rnn_tensorized,
            label="VCA with Tensorized RNN",
            marker="o",
            linestyle="-",
            color="black"
        )
        plt.plot(
            steps,
            average_energies_rnn,
            label="VCA with Simple RNN",
            marker="o",
            linestyle="-",
            color="blue"
        )
        # plt.plot(
        #     steps,
        #     average_energies_rnn_bn,
        #     label="VCA with RNN + Batch Normalization",
        #     marker="o",
        #     linestyle="-",
        #     color="red"
        # )
        plt.plot(
            steps,
            average_energies_MCMC,
            label="SA",
            marker="o",
            linestyle="-",
            color="orange"
        )
        
        plt.xlabel("Transition Steps")
        plt.ylabel("Average Energy")
        plt.title("Average Energy of Solutions versus Transition Steps - " + model_type)

        plt.legend()

        plt.savefig(save_dir + model_type + "_Average_Energy_Plot_TEST.png")
