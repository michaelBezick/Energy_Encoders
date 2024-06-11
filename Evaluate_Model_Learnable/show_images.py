import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

from Functions import (
    BVAE,
    clamp_output,
    compute_pearson_correlation,
    expand_output,
    get_annealing_vectors,
    get_energy_fn,
    get_folder_path_from_model_path,
    get_list_of_models,
    get_model_name_and_type,
    get_sampling_vars,
    load_energy_functions,
    load_FOM_model,
    load_from_checkpoint,
    threshold,
)
from Modules.Energy_Encoder_Classes import CorrelationalLoss

device = "cuda"
save_images = True

clamp, threshold = threshold()

Blume_Capel_energy, Potts_energy, QUBO_energy = load_energy_functions(device)
energy_fn_list = [QUBO_energy, Potts_energy, Blume_Capel_energy]

energy_loss_fn = CorrelationalLoss()

FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")

models_list = get_list_of_models()

log_dir = ""

"""
# load QIOTE vectors
optimal_vectors_list_qiote = None
with open("optimal_vectors.pkl", "rb") as file:
    optimal_vector_list_qiote = pickle.load(file)

"""


Blume_Capel_vectors, Potts_vectors, QUBO_vectors = get_annealing_vectors()
optimal_vectors = [Blume_Capel_vectors, Potts_vectors, QUBO_vectors]

"""All neural annealing for now"""
num_experiment = 1

# if num_experiment == 0:
#     experiment_type = "QIOTE"
#     num_iters = 10
#     num_experiment += 1
#     bias = 0
# else:
#     experiment_type = "Neural"
#     num_iters = optimal_vector_list.size()[0] // 100
#     bias = 0

largest_FOM_global = 0
for model_dir in tqdm(models_list):
    energies = []
    FOM_global = []

    model_name, model_type = get_model_name_and_type(model_dir)

    energy_fn = None

    if model_name == "Blume-Capel":
        optimal_vector_list = optimal_vectors[0]
        continue
    elif model_name == "Potts":
        optimal_vector_list = optimal_vectors[1]
        continue
    else:
        optimal_vector_list = optimal_vectors[2]

    print(optimal_vector_list.size())
    num_iters = optimal_vector_list.size()[0] // 100
    print(num_iters)
    energy_fn = get_energy_fn(model_name, energy_fn_list)

    model = BVAE(energy_fn, energy_loss_fn, h_dim=128, model_type=model_type).to(device)
    model = load_from_checkpoint(model, model_dir)
    model = model.eval()

    num_logits, scale = get_sampling_vars(model)

    zero_tensor = torch.zeros([100, 64])
    FOM_measurements = []
    largest_FOM = 0

    with torch.no_grad():
        for iters in range(num_iters):
            zero_tensor = optimal_vector_list[iters * 100 : (iters + 1) * 100]

            vectors = zero_tensor.cuda()

            vectors_energies = energy_fn(vectors)
            numpy_energies = vectors_energies.detach().cpu().numpy()
            energies.extend(numpy_energies)

            output = model.vae.decode(vectors)
            output_expanded = expand_output(output)

            if clamp:
                output_expanded = clamp_output(output_expanded, threshold)

            FOM = FOM_calculator(
                torch.permute(output_expanded.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy()
            )

            FOM_measurements.extend(FOM.numpy().flatten().tolist())
            FOM_global.extend(FOM.numpy().flatten().tolist())

            if np.max(np.array(FOM_measurements)) > largest_FOM:
                largest_FOM = np.max(np.array(FOM_measurements))

            grid = torchvision.utils.make_grid(output_expanded[0:16, :, :].cpu(), nrow=4)

            model_folder_path = get_folder_path_from_model_path(model_dir)
            log_dir = model_folder_path
            print(log_dir)

            if save_images:
                save_image(grid, log_dir + "generated.png", dpi=600)
                exit()

        FOM_measurements = np.array(FOM_measurements)
        average = np.mean(FOM_measurements)

        if largest_FOM > largest_FOM_global:
            largest_FOM_global = largest_FOM

        # calculating pearson correlation
        pearson_correlation = compute_pearson_correlation(FOM_global, energies)

        with open(log_dir + "/FOM_data.txt", "w") as file:
            file.write(f"Average FOM: {average}\n")
            file.write(f"Max FOM: {largest_FOM}\n")
            file.write(f"pearson_correlation: {pearson_correlation}")

        plt.figure()
        plt.scatter(energies, FOM_global)
        plt.xlabel("Energy of vectors")
        plt.ylabel("FOM of samples")
        plt.title("Sampled correlation")
        plt.savefig(log_dir + "/SampledCorrelation.png")


with open("Experiment_Summary.txt", "w") as file:
    file.write(f"Experiment max FOM: {largest_FOM_global}")
