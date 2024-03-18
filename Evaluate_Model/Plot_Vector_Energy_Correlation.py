import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Functions import get_folder_path_from_model_path, get_list_of_models, get_title_from_model_path, \
                        load_dataset, get_energy_fn, load_energy_functions, BVAE
import torch
from tqdm import tqdm

"""
Code requires that each model be stored as its own directory within ./Models as soley a .ckpt file.
For example, ./Models/Blume-Capel/0_MCMC_0,1_temp/good.ckpt
"""

batch_size = 100
device = "cuda"

models_list = get_list_of_models()

Blume_Capel_energy, Potts_energy, QUBO_energy = load_energy_functions(device)

energy_fn_list = [QUBO_energy, Potts_energy, Blume_Capel_energy]
model = BVAE(None, None, h_dim = 128)

for model_dir in tqdm(models_list):

    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(device)

    energy_fn = get_energy_fn(model, energy_fn_list)

    #need to plot FOM versus energy
    dataset = load_dataset("../Files/top_0.npy")

    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, drop_last = True)

    FOMs = []
    energies = []

    model = model.eval()
    with torch.no_grad():
        for data in train_loader:
            x, FOM = data
            x = x.to(device)
            vectors = model.vae.encode(x)
            print(vectors.size())
            exit()
            energy = energy_fn(vectors)

            FOMs.append(FOM)
            energies.append(energy)

    plt.figure()
    plt.scatter(energies, FOMs)
    plot_title = get_title_from_model_path(model_dir)
    plt.title(plot_title)

    model_folder_path = get_folder_path_from_model_path(model_dir)
    plt.savefig(model_folder_path)
