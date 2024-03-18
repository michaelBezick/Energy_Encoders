import torch
import pytorch_lightning as pl
import torchvision
import torch.optim
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import argparse
import tensorflow as tf
import polytensor
from enum import Enum
from Modules.Energy_Encoder_Classes import BVAE

class Model_Type(Enum):
    QUBO = 1
    PUBO = 2
    ISING = 3
    BLUME_CAPEL = 4
    POTTS = 5

def blume_capel_scale(x):
    return x - 1
def ising_scale(x):
    return 2 * x - 1
def no_scale(x):
    return x

class Potts_Energy_Fn(nn.Module):
    def __init__(self, interactions, batch_size = 100):
        super().__init__()
        self.interactions = interactions
        self.batch_size = batch_size

    def dirac_delta(self, x, y):
        return (1 - x) * (1 - y) + (x * y)

    def forward(self, vector):
        vector = vector.unsqueeze(1)
        dirac_delta_terms = self.dirac_delta(vector, torch.transpose(vector, 1, 2))
        dirac_delta_terms = torch.triu(dirac_delta_terms)
        energy_matrix = torch.mul(dirac_delta_terms, self.interactions)
        energy_matrix = energy_matrix.view(self.batch_size, -1)

        return torch.sum(energy_matrix, dim = 1)

def load_energy_functions(device):
    num_vars = 64

    num_per_degree = [num_vars, num_vars * (num_vars - 1) // 2]
    sample_fn = lambda: torch.randn(1, device='cuda')
    terms = polytensor.generators.coeffPUBORandomSampler(
            n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
            )
    terms = polytensor.generators.denseFromSparse(terms, num_vars)

    Blume_Capel_model = polytensor.DensePolynomial(terms)
    QUBO_model = polytensor.DensePolynomial(terms)

    Blume_Capel_coeff = torch.load("./Energy_Functions/Blume-Capel_energy_fn_coefficients.pt")
    Potts_coeff = torch.load("./Energy_Functions/Potts_energy_fn_coefficients.pt")
    QUBO_coeff = torch.load("./Energy_Functions/QUBO_energy_fn_coefficients.pt")

    Blume_Capel_model.coefficients = Blume_Capel_coeff
    Potts_model = Potts_Energy_Fn(Potts_coeff)
    QUBO_model.coefficients = QUBO_coeff

    return Blume_Capel_model, Potts_model, QUBO_model


def expand_output(tensor: torch.Tensor):
    x = torch.zeros([100, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x

def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

def load_FOM_model(model_path, weights_path):
    with open(model_path, 'r') as file:
        data = file.read()

    FOM_calculator = tf.keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator

def threshold():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0., help="threshold to clamp values")

    args = parser.parse_args()
    threshold = args.threshold

    if not threshold == 0.0:
        clamp = True
    else:
        clamp = False

    return clamp, threshold


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

    labels = torch.from_numpy(np.squeeze(np.load('../Files/FOM_labels.npy')))

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
