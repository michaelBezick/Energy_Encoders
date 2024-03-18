import torch
import torchvision
import pickle
from torchvision.utils import save_image
import os
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
from Functions import threshold, load_FOM_model, expand_output, clamp_output

iteration_list = [0, 1, 3, 5, 10]

clamp, threshold = threshold()

FOM_evaluator = load_FOM_model("VGGnet.json", "VGGnet_weights.h5")

QUBO_energy = torch.load("./Energy_Functions/QUBO_energy_fn")
Potts_energy = torch.load("./Energy_Functions/Potts_energy_fn")
Blume_Capel_energy = torch.load("./Energy_Functions/Blume-Capel_energy_fn")

FOM_dict = {}

max_FOM_neural = 0
optimal_vectors_list_qiote = None
with open("optimal_vectors.pkl", "rb") as file:
    optimal_vector_list_qiote = pickle.load(file)

"""
Changed to neural annealing
"""

optimal_vector_list_neural = torch.load("./neural_annealing_vectors.pt")
print(optimal_vector_list_neural.size())

optimal_vectors = [optimal_vector_list_qiote, optimal_vector_list_neural]

num_experiment = 0

for optimal_vector_list in optimal_vectors:

    experiment_type = ""

    if num_experiment == 0:
        experiment_type = "QIOTE" 
        num_iters = 10
        num_experiment += 1
        bias = 0
    else:
        experiment_type = "Neural"
        num_iters = 300
        bias = 0

    for num_MCMC_iterations in iteration_list:
        energies = []
        FOM_global = []

        checkpoint_path1 = f"./checkpoints/Pearson/{num_MCMC_iterations}_MCMC_temp_0,1"

        checkpoint_path2 = os.listdir(checkpoint_path1)[-1]
        checkpoint_path = os.path.join(checkpoint_path1, checkpoint_path2)
        checkpoint_path = checkpoint_path + "/checkpoints/"
        file_checkpoint = os.listdir(checkpoint_path)[0]
        checkpoint_path = os.path.join(checkpoint_path, file_checkpoint)
        print(checkpoint_path)

        if not os.path.isdir(f"./Optimal_Images/Pearson/{experiment_type}/{num_MCMC_iterations}_iterations"):
            os.mkdir(f"./Optimal_Images/Pearson/{experiment_type}/{num_MCMC_iterations}_iterations")

        epochs = 1000
        batch_size = 100
        reconstruction_term = 0
        perceptual_term = 0
        kl_term = 0
        adv_term = 0
        energy_loss_term = 0
        lr = 1e-3
        latent_vector_dim = 64
        QUBO_matrix = torch.load("./pearson_QUBO_matrix.pt")
        QUBO_matrix = QUBO_matrix.cuda()

        vae_gan = VAE_GAN(reconstruction_term, perceptual_term, 
                      kl_term, adv_term, energy_loss_term, 1, 
                      128, lr = lr, num_MCMC_iterations=num_MCMC_iterations, 
                      latent_vector_dim=latent_vector_dim, batch_size=batch_size,
                      QUBO_matrix=QUBO_matrix)

        checkpoint = torch.load(checkpoint_path)

        vae_gan.load_state_dict(checkpoint['state_dict'])

        vae_gan = vae_gan.cuda()

        vae_gan = vae_gan.eval()

        zero_tensor = torch.zeros([100, 64])
        FOM_measurements = []
        largest_FOM = 0
        with torch.no_grad():
            for iters in range(num_iters):
                zero_tensor = optimal_vector_list[iters * 100:(iters+1)*100]
                #for i in range(100):
                #    zero_tensor[i, :] = optimal_vector_list[iters * 100 + i + bias * 100]

                vectors = zero_tensor.cuda()
                
                vectors_energies = quboEnergy(vectors, QUBO_matrix)
                numpy_energies = vectors_energies.detach().cpu().numpy()
                for energy in numpy_energies:
                    energies.append(energy)


                output = vae_gan.vae.decode(vectors)
                output_expanded = expand_output(output)

                if clamp:
                    output_expanded = clamp_output(output_expanded, threshold)

                FOM = FOM_calculator(torch.permute(output_expanded.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy())
                FOM_measurements.append(FOM)

                for merit in FOM:
                    FOM_global.append(merit)

                if np.max(np.array(FOM_measurements)) > largest_FOM:
                    largest_FOM = np.max(np.array(FOM_measurements)) 


                grid = torchvision.utils.make_grid(output_expanded.cpu())

                log_dir = f"Optimal_Images/Pearson/{experiment_type}/{num_MCMC_iterations}_iterations" + f"/{iters}.png"
                print(log_dir)

                #save_image(grid, log_dir)
            
            FOM_measurements = np.array(FOM_measurements)
            average = np.mean(FOM_measurements)
            FOM_dict[num_MCMC_iterations] = largest_FOM
            with open(f"Optimal_Images/Pearson/{experiment_type}/{num_MCMC_iterations}_iterations" + "/FOM_data.txt", "w") as file:
                file.write(f"Average FOM: {average}\n")
                file.write(f"Max FOM: {largest_FOM}")

            plt.figure()
            plt.scatter(energies, FOM_global)
            plt.xlabel("Energy of vectors")
            plt.ylabel("FOM of samples")
            plt.title("Sampled correlation")
            plt.savefig(f"Optimal_Images/Pearson/{experiment_type}/{num_MCMC_iterations}_iterations" + "/SampledCorrelation.png")
