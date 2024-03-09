import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import os

from Evaluate_Model.Plot_Vector_Energy_Correlation import Blume_Capel_energy, Potts_energy, QUBO_energy

class LabeledDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image[:, 0:32, 0:32], label

def load_dataset(path):
    dataset = np.expand_dims(np.load(path), 1)
    normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
        
    normalizedDataset = np.multiply(normalizedDataset, 2) - 1

    normalizedDataset = normalizedDataset.astype(np.float32)

    dataset = torch.from_numpy(normalizedDataset)

    labels = torch.from_numpy(np.squeeze(np.load('FOM_labels.npy')))

    labeled_dataset = LabeledDataset(dataset, labels)
    return labeled_dataset

def get_list_of_models():
    path_list = os.listdir("./Models/") 
    models_list = []
    for path in path_list:
        path = "./Models/" + path + "/"
        model_list = os.listdir(path)
        for model_path in model_list:
            model_path = path + model_path #currently have the folder name, which is a unique ID for the model
            model_path = model_path + "/" + os.listdir(model_path)[0] #getting the unique checkpoint to each model
            models_list.append(model_path)

    return models_list

def get_folder_path_from_model_path(model_path):
    model_dir_split = model_path.split("/")[:4]
    model_folder_path = '/'.join([str(item) for item in model_dir_split]) + "/"
    return model_folder_path

def get_title_from_model_path(model_path):
    plot_title = model_path.split("/")[3]
    return plot_title

def get_energy_fn(model, energy_fn_list):

    QUBO_energy = energy_fn_list[0]
    Potts_energy = energy_fn_list[1]
    Blume_Capel_energy = energy_fn_list[2]

    if model.model_type == 'QUBO':
        energy_fn = QUBO_energy
    elif model.model_type == 'Potts':
        energy_fn = Potts_energy
    else:
        energy_fn = Blume_Capel_energy

    return energy_fn
