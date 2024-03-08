import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Functions import load_dataset
import torch
import os

batch_size = 100
device = "cuda"

path_list = os.listdir("./Models/") 
models_list = []
for path in path_list:
    path = "./Models/" + path + "/"
    model_list = os.listdir(path)
    for model_path in model_list:
        model_path = path + model_path #currently have the folder name, which is a unique ID for the model
        model_path = model_path + "/" + os.listdir(model_path)[0] #getting the unique checkpoint to each model
        models_list.append(model_path)

for model_dir in models_list:
    model_dir_split = model_dir.split("/")[:4]
    model_folder_path = '/'.join([str(item) for item in model_dir_split]) + "/"

#print(models_list)
#for model_path in models_list:
#    print(model_path.split("/")[3]) #gets the unique model name now

QUBO_energy = torch.load("./Energy_Functions/QUBO_energy_fn")
Potts_energy = torch.load("./Energy_Functions/Potts_energy_fn")
Blume_Capel_energy = torch.load("./Energy_Functions/Blume-Capel_energy_fn")

for model_dir in models_list:

    model = torch.load(model_dir)
    model = model.to(device)
    if model.model_type == 'QUBO':
        energy_fn = QUBO_energy
    elif model.model_type == 'Potts':
        energy_fn = Potts_energy
    else:
        energy_fn = Blume_Capel_energy

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
    plot_title = model_dir.split("/")[3]
    plt.title(plot_title)

    model_dir_split = model_dir.split("/")[:4]
    model_folder_path = '/'.join([str(item) for item in model_dir_split]) + "/"
    plt.savefig(model_folder_path)
