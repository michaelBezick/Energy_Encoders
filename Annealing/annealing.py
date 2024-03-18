import torch
import time
import matplotlib.pyplot as plt
from annealing_classes import RNN_Concat, Variational_Free_Energy
from annealing_functions import load_energy_functions

device = 'cuda'
epochs = 1000
lr = 5e-4
batch_size = 100
warmup_steps = 0
temperature = 1
N_gradient_descent = 1
N_samples = 50
log_step_size = 10
min_energy_repeat_threshold = 400
vector_length = 64

print_vector = False
plot = True
save_vectors = True

#experiment with normalizing energy functions
Blume_Capel_energy, Potts_energy, QUBO_energy = load_energy_functions(device) #loads from Evaluate Model
energy_fn_list = [Blume_Capel_energy, Potts_energy, QUBO_energy]

for experiment_number, energy_fn in enumerate(energy_fn_list):
    if experiment_number == 0:
        continue
    elif experiment_number == 1:
        continue
    energy_fn = energy_fn.to(device)
    energy_loss = Variational_Free_Energy(energy_fn, N_samples = N_samples, batch_size = batch_size)
    energy_loss = energy_loss.to(device)
    rnn = RNN_Concat().to(device)

    optimizer = torch.optim.Adam(params = rnn.parameters(), lr = lr)

    initial_vector = torch.bernoulli(torch.ones(batch_size, vector_length) * 0.5).to(device)
    sigma = initial_vector

    average_energies = []
    temperatures = []

    min_energy = 0
    best_vector = torch.zeros(vector_length).to(device)

    start_time = time.time()

    unique_vector_list = []
    unique_vector_set = set()

    #sigma is sampled
    #sigma_hat is probabililties
    for i in range(warmup_steps):
        sigma_hat = rnn(sigma)

        loss = energy_loss(sigma_hat, temperature)
        if i % log_step_size == 0:
            print(f"Average Energy: {torch.mean(energy_fn(sigma))}")
            print(f"Loss: {loss}")
            print(f"i: {i}")
            average_energies.append(torch.mean(energy_fn(sigma)).item())
            temperatures.append(temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #move to next state after n gradient descent steps
        if i % N_gradient_descent == 0:
            sigma = torch.bernoulli(sigma_hat)

    min_energy_repeats = 0
    epoch = 0
    for epoch in range(epochs):
        temperature -= 1 / epochs
        if temperature <= 0:
            temperature = 0
        sigma_hat = rnn(sigma)

        loss = energy_loss(sigma_hat, temperature)

        #adding vectors to unique vector set
        list_of_vectors = torch.bernoulli(sigma).tolist()
        for vector in list_of_vectors:
            vector_tuple = tuple(vector)
            if vector_tuple not in unique_vector_set:
                unique_vector_set.add(vector_tuple)
                unique_vector_list.append(vector)

        #finding min energy, with consideration of repeats
        min_energy_repeats += 1

        if torch.min(energy_fn(sigma)) < min_energy:
            min_energy_repeats = 0
            min_energy = torch.min(energy_fn(sigma))
            index = torch.argmin(energy_fn(sigma))
            best_vector = sigma[index, :]

        if min_energy_repeats > min_energy_repeat_threshold:
            break

        if epoch % log_step_size == 0:
            print(f"Epoch: {epoch}")
            print(f"Min Energy: {min_energy}")
            print(f"Loss: {loss}")
            if (print_vector):
                print(f"{best_vector}")
            average_energies.append(torch.mean(energy_fn(sigma)).item())
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
        save_dir = "./Blume-Capel/"
    elif experiment_number == 1:
        save_dir = "./Potts/"
    else:
        save_dir = "./QUBO/"


    print(f"Elapsed time: {elapsed_time}")
    print(f"Time per epoch: {time_per_epoch}")
    sigma = torch.tensor(unique_vector_list)
    print(sigma.size())
    if (save_vectors):
        torch.save(sigma, save_dir + "neural_annealing_vectors.pt")

    if plot:
        total_steps = epoch + warmup_steps
        steps = list(range(0, total_steps, log_step_size))

        plt.figure()

        plt.plot(steps, average_energies, label="Average Variational Classical Annealing Solution Energy", marker='o', linestyle='-')
        plt.xlabel('Transition Steps')
        plt.ylabel('Average Energy')
        #plt.axhline(y=-391, color='r', label='Simulated Annealing Min Solution')
        plt.title('Average Energy of Solutions versus Transition Steps')

        plt.legend()

        plt.savefig(save_dir + "Average_Energy_Plot.png")
