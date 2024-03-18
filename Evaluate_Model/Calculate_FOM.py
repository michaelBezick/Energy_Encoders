import torch
import torchvision
import pickle
from torchvision.utils import save_image
import torch
import numpy as np
import matplotlib.pyplot as plt
from Functions import load_from_checkpoint, threshold, load_FOM_model, expand_output, clamp_output
import torch
import matplotlib.pyplot as plt
from Modules.Energy_Encoder_Classes import CorrelationalLoss
from Functions import get_folder_path_from_model_path, get_list_of_models,\
                        get_energy_fn, load_energy_functions, BVAE, \
                        get_sampling_vars, get_model_name_and_type
import torch
from tqdm import tqdm

device = "cuda"
save_images = True

clamp, threshold = threshold()

Blume_Capel_energy, Potts_energy, QUBO_energy = load_energy_functions(device)
energy_fn_list = [QUBO_energy, Potts_energy, Blume_Capel_energy]

energy_loss_fn = CorrelationalLoss()

FOM_calculator = load_FOM_model("VGGnet.json", "VGGnet_weights.h5")

models_list = get_list_of_models()

FOM_dict = {}
log_dir = ""

#load QIOTE vectors
optimal_vectors_list_qiote = None
with open("optimal_vectors.pkl", "rb") as file:
    optimal_vector_list_qiote = pickle.load(file)

#load Neural Annealing Vectors
max_FOM_neural = 0
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
        num_iters = optimal_vector_list_neural.size() // 100
        bias = 0

    for model_dir in tqdm(models_list):
        energies = []
        FOM_global = []

        model_name, model_type = get_model_name_and_type(model_dir)

        energy_fn = get_energy_fn(model_name, energy_fn_list)

        model = BVAE(energy_fn, energy_loss_fn, h_dim = 128, model_type=model_type).to(device)
        model = load_from_checkpoint(model, model_dir)
        model = model.eval()

        num_logits, scale = get_sampling_vars(model)

        zero_tensor = torch.zeros([100, 64])
        FOM_measurements = []
        largest_FOM = 0
        with torch.no_grad():
            for iters in range(num_iters):
                zero_tensor = optimal_vector_list[iters * 100:(iters+1)*100]

                vectors = zero_tensor.cuda()
                
                vectors_energies = energy_fn(vectors)
                numpy_energies = vectors_energies.detach().cpu().numpy()
                energies.extend(numpy_energies)

                output = model.vae.decode(vectors)
                output_expanded = expand_output(output)

                if clamp:
                    output_expanded = clamp_output(output_expanded, threshold)

                FOM = FOM_calculator(torch.permute(output_expanded.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy())
                FOM_measurements.append(FOM)
                FOM_global.extend(FOM)

                if np.max(np.array(FOM_measurements)) > largest_FOM:
                    largest_FOM = np.max(np.array(FOM_measurements)) 

                grid = torchvision.utils.make_grid(output_expanded.cpu())

                model_folder_path = get_folder_path_from_model_path(model_dir)
                log_dir = model_folder_path

                if save_images:
                    save_image(grid, log_dir)
            
            FOM_measurements = np.array(FOM_measurements)
            average = np.mean(FOM_measurements)

            """Not sure why I did this"""
            #FOM_dict[num_MCMC_iterations] = largest_FOM

            with open(log_dir + "/FOM_data.txt", "w") as file:
                file.write(f"Average FOM: {average}\n")
                file.write(f"Max FOM: {largest_FOM}")

            plt.figure()
            plt.scatter(energies, FOM_global)
            plt.xlabel("Energy of vectors")
            plt.ylabel("FOM of samples")
            plt.title("Sampled correlation")
            plt.savefig(log_dir + "/SampledCorrelation.png")
