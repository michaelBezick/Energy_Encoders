import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Energy_Functions.Functions import get_folder_path_from_model_path, get_list_of_models, get_title_from_model_path, load_dataset, get_energy_fn
import torch

"""
Code requires that each model be stored as its own directory within ./Models as soley a .ckpt file.
For example, ./Models/Blume-Capel/0_MCMC_0,1_temp/good.ckpt
"""

batch_size = 100
device = "cuda"

models_list = get_list_of_models()

QUBO_energy = torch.load("./Energy_Functions/QUBO_energy_fn")
Potts_energy = torch.load("./Energy_Functions/Potts_energy_fn")
Blume_Capel_energy = torch.load("./Energy_Functions/Blume-Capel_energy_fn")

energy_fn_list = [QUBO_energy, Potts_energy, Blume_Capel_energy]

for model_dir in models_list:

    model = torch.load(model_dir)
    model = model.to(device)

    energy_fn = get_energy_fn(model, energy_fn_list)

    #need to plot FOM versus energy
    dataset = load_dataset("top_0.npy")

    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, drop_last = True)

    FOMs = []
    energies = []

    for data in train_loader:
        data = data.to(device)
        x, FOM = data
        vectors = model.encode(x)
        energy = energy_fn(vectors)

        FOMs.append(FOM)
        energies.append(energy)

    plt.figure()
    plt.scatter(energies, FOMs)
    plot_title = get_title_from_model_path(model_dir)
    plt.title(plot_title)

    model_folder_path = get_folder_path_from_model_path(model_dir)
    plt.savefig(model_folder_path)
