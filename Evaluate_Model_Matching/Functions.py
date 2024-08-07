import argparse
import os

import numpy as np
import polytensor
import pytorch_lightning as pl
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision
from torch.utils.data import Dataset

from Energy_Encoder_Classes import BVAE, Model_Type


def compute_pearson_correlation(x_FOM, y_Energy):

    x_FOM = torch.Tensor(np.array(x_FOM))
    y_Energy = torch.Tensor(np.array(y_Energy))

    # x should be a vector of length n
    # y should be a vector of length n

    x_FOM = torch.squeeze(x_FOM)
    y_Energy = torch.squeeze(y_Energy)
    x_mean = torch.mean(x_FOM)
    y_mean = torch.mean(y_Energy)

    x_deviation_from_mean = x_FOM - x_mean
    y_deviation_from_mean = y_Energy - y_mean

    covariance = torch.einsum("i,i->", x_deviation_from_mean, y_deviation_from_mean)

    if covariance == 0:
        print("COVARIANCE 0")
        exit()

    std_dev_x = torch.sqrt(
        torch.einsum("i,i->", x_deviation_from_mean, x_deviation_from_mean)
    )
    if std_dev_x == 0:
        print("std_dev_x 0")
        exit()
    std_dev_y = torch.sqrt(
        torch.einsum("i,i->", y_deviation_from_mean, y_deviation_from_mean)
    )
    if std_dev_y == 0:
        print("std_dev_y 0")
        exit()

    pearson_correlation_coefficient = covariance / (std_dev_x * std_dev_y)

    return pearson_correlation_coefficient


def get_annealing_vectors():
    second = torch.load(
        "../Annealing_Matching/Models/QUBO_order_2/neural_annealing_vectors.pt"
    )
    third = torch.load(
        "../Annealing_Matching/Models/QUBO_order_3/neural_annealing_vectors.pt"
    )
    fourth = torch.load(
        "../Annealing_Matching/Models/QUBO_order_4/neural_annealing_vectors.pt"
    )

    return second, third, fourth


def load_from_checkpoint(model, model_dir):
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint["state_dict"])

    return model


def scale_vector_copy_gradient(x, probabilities, scale):
    """
    x in index format -> x in scaled format with gradient
    """
    x = F.one_hot(x)
    copied_grad = (x - probabilities).detach() + probabilities
    return torch.einsum("ijk,k->ij", copied_grad, scale)


def blume_capel_scale(x):
    return x - 1


def ising_scale(x):
    return 2 * x - 1


def no_scale(x):
    return x


def get_sampling_vars(model):
    if model.model_type == "QUBO":
        num_logits = 2
        scale = torch.Tensor([0.0, 1.0])
    elif model.model_type == "PUBO":
        num_logits = 2
        scale = torch.Tensor([0.0, 1.0])
    elif model.model_type == "Blume-Capel":
        num_logits = 3
        scale = torch.Tensor([-1.0, 0.0, 1.0])
    elif model.model_type == "Potts":
        num_logits = 2
        scale = torch.Tensor([0.0, 1.0])
    else:
        raise ValueError("Model does not exist!")

    return num_logits, scale


class Potts_Energy_Fn(nn.Module):
    def __init__(self, interactions, batch_size=100):
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

        return torch.sum(energy_matrix, dim=1)


def load_energy_functions(device):
    num_vars = 64

    num_per_degree = [num_vars, num_vars * (num_vars - 1) // 2]
    sample_fn = lambda: torch.randn(1, device="cuda")
    terms = polytensor.generators.coeffPUBORandomSampler(
        n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
    )
    terms = polytensor.generators.denseFromSparse(terms, num_vars)

    Blume_Capel_model = polytensor.DensePolynomial(terms)
    QUBO_model = polytensor.DensePolynomial(terms)

    Blume_Capel_coeff = torch.load(
        "./Energy_Functions/Blume-Capel_energy_fn_coefficients.pt"
    ).to(device)
    Potts_coeff = torch.load("./Energy_Functions/Potts_energy_fn_coefficients.pt").to(
        device
    )
    QUBO_coeff = torch.load("./Energy_Functions/QUBO_energy_fn_coefficients.pt").to(
        device
    )

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
    with open(model_path, "r") as file:
        data = file.read()

    FOM_calculator = tf.keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator


def threshold():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=0.0, help="threshold to clamp values"
    )

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
    normalizedDataset = (dataset - np.min(dataset)) / (
        np.max(dataset) - np.min(dataset)
    )

    normalizedDataset = np.multiply(normalizedDataset, 2) - 1

    normalizedDataset = normalizedDataset.astype(np.float32)

    dataset = torch.from_numpy(normalizedDataset)

    labels = torch.from_numpy(np.squeeze(np.load("../Files/FOM_labels.npy")))

    labeled_dataset = LabeledDataset(dataset, labels)
    return labeled_dataset


# def get_list_of_models():
#     path_list = os.listdir("./Models/")
#     models_list = []
#     for path in path_list:
#         path = "./Models/" + path + "/"
#         model_list = os.listdir(path)
#         for model_path in model_list:
#             model_path = (
#                 path + model_path
#             )  # currently have the folder name, which is a unique ID for the model
#             model_path = (
#                 model_path + "/" + os.listdir(model_path)[0]
#             )  # getting the unique checkpoint to each model
#             models_list.append(model_path)
#
#     return models_list


def get_list_of_models():
    path_list = os.listdir("./Models/")
    models_list = []
    for path in path_list:
        path = "./Models/" + path + "/"
        model_list = os.listdir(path)
        for model_path in model_list:
            if "old_files" in model_path:
                continue
            model_path = path + model_path

            # need this because checkpoint file is not sole file
            files = os.listdir(model_path)
            checkpoint_path = ""
            for potential_checkpoint in files:
                if ".ckpt" in potential_checkpoint:
                    checkpoint_path = potential_checkpoint
                    break

            model_path = model_path + "/" + checkpoint_path

            models_list.append(model_path)

    return models_list


def get_folder_path_from_model_path(model_path):
    model_dir_split = model_path.split("/")[:4]
    model_folder_path = "/".join([str(item) for item in model_dir_split]) + "/"
    return model_folder_path


def get_title_from_model_path(model_path):
    plot_title = model_path.split("/")[3]
    return plot_title


def get_energy_fn(model_name, energy_fn_list):

    QUBO_energy = energy_fn_list[0]
    Potts_energy = energy_fn_list[1]
    Blume_Capel_energy = energy_fn_list[2]

    if model_name == "QUBO":
        energy_fn = QUBO_energy
    elif model_name == "Potts":
        energy_fn = Potts_energy
    else:
        energy_fn = Blume_Capel_energy

    return energy_fn
