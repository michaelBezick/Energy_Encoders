from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import polytensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

from Energy_Encoder_Classes import VAE

scale = torch.Tensor([0.0, 1.0])


def calc_norm(terms):
    sum_of_squares = torch.zeros(1)
    for term in terms:
        sum_of_squares = sum_of_squares + torch.sum(term**2)

    # sum_of_squares = torch.sum(terms[0] ** 2) + torch.sum(terms[1] ** 2)
    return torch.sqrt(sum_of_squares)


def expand_output(tensor: torch.Tensor, batch_size):
    x = torch.zeros([batch_size, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x


def scale_vector_copy_gradient(x, probabilities):
    """
    x in index format -> x in scaled format with gradient
    """
    x = F.one_hot(x, num_logits)
    copied_grad = (x - probabilities).detach() + probabilities
    return torch.einsum("ijk,k->ij", copied_grad, scale)


def mean_normalize(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


class Model_Type(Enum):
    QUBO = 1
    PUBO = 2
    ISING = 3
    BLUME_CAPEL = 4
    POTTS = 5


class CorrelationalLoss:
    """
    Idea:
    1. want the correlation to be closest to -1
    2. want the average energy to be minimized
    3. want the slope of the linear correlation to be minimized
    """

    def __init__(self, correlation_weight=1.0, energy_weight=1.0, slope_weight=1.0):

        self.correlation_weight = correlation_weight
        self.energy_weight = energy_weight
        self.slope_weight = slope_weight

        self.covariance = 0
        self.std_dev_x = 0

        self.correlation_loss = 0
        self.average_energy_loss = 0
        self.slope_loss = 0

        self.correlation = 0
        self.average_energy = 0
        self.slope = 0

    def __call__(self, x_FOM, y_Energy):
        pearson_correlation_coefficient = self.compute_pearson_correlation(
            x_FOM, y_Energy
        )
        average_energy = self.compute_average_energy(y_Energy)
        slope = self.compute_slope()

        self.correlation = pearson_correlation_coefficient
        self.average_energy = average_energy
        self.slope = slope

        x = pearson_correlation_coefficient
        correlation_loss = torch.log((0.5 * (x + 1)) / (1 - 0.5 * (x + 1)))

        average_energy_loss = average_energy

        slope_loss = slope

        loss_combined = (
            self.correlation_weight * correlation_loss
            + self.energy_weight * average_energy_loss
            + self.slope_weight * slope_loss
        )

        self.correlation_loss = correlation_loss * self.correlation_weight
        self.average_energy_loss = average_energy_loss * self.energy_weight
        self.slope_loss = slope_loss * self.slope_weight

        return loss_combined

    def print_info(self):
        print(
            f"correlation: {self.correlation}\taverage_energy: {self.average_energy}\tslope: {self.slope}"
        )

    def print_losses(self):
        print(
            f"correlation_loss: {self.correlation_loss}\tenergy_loss: {self.average_energy_loss}\tslope_loss: {self.slope_loss}"
        )

    def compute_slope(self):
        return self.covariance / self.std_dev_x

    def compute_average_energy(self, y_Energy):
        return torch.mean(y_Energy)

    def compute_pearson_correlation(self, x_FOM, y_Energy):

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

        self.covariance = covariance
        self.std_dev_x = std_dev_x

        return pearson_correlation_coefficient


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


def quboEnergy(x, H):
    """
    Computes the energy for the specified Quadratic Unconstrained Binary Optimization (QUBO) system.

    Parameters:
        x (torch.Tensor) : Tensor of shape (batch_size, num_dim) representing the configuration of the system.
        H (torch.Tensor) : Tensor of shape (batch_size, num_dim, num_dim) representing the QUBO matrix.

    Returns:
        torch.Tensor : The energy for each configuration in the batch.
    """
    if len(x.shape) == 1 and len(H.shape) == 2:
        return torch.einsum("i,ij,j->", x, H, x)
    elif len(x.shape) == 2 and len(H.shape) == 3:
        return torch.einsum("bi,bij,bj->b", x, H, x)
    elif len(x.shape) == 2 and len(H.shape) == 2:
        return torch.einsum("bi,ij,bj->b", x, H, x)
    else:
        raise ValueError(
            "Invalid shapes for x and H. x must be of shape (batch_size, num_dim) and H must be of shape (batch_size, num_dim, num_dim)."
        )


class Energy_Fn(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(64, 64))
        self.loss_fn = nn.MSELoss()

    def forward(self, vectors):
        vectors = torch.squeeze(vectors)

        energy = quboEnergy(vectors, self.coefficients)

        return energy


def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))


dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

dataset = clamp_output(dataset, 0.5)

labels = torch.load("../Files/FOM_labels_new.pt")

labeled_dataset = LabeledDataset(dataset, labels)

batch_size = 200

train_loader = DataLoader(
    labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

# energyFn = Energy_Fn()
num_vars = 64

num_per_degree = [num_vars, num_vars * (num_vars - 1) // 2]
sample_fn = lambda: torch.randn(1, device="cuda")
terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

terms = polytensor.generators.denseFromSparse(terms, num_vars)
norm = calc_norm(terms)
terms[0] = terms[0] / norm
terms[1] = terms[1] / norm

energyFn = polytensor.DensePolynomial(terms)

energy_loss_fn = CorrelationalLoss(10, 0.01, 0.001)
lr = 1e-3
vae = VAE(batch_size=batch_size, num_logits=2)

energy_optimizer = torch.optim.Adam(params=energyFn.parameters(), lr=lr)
model_optimizer = torch.optim.Adam(params=vae.parameters(), lr=lr)

num_logits = 2
latent_vector_dim = 64
sampler = torch.multinomial

i = 0
for epoch in range(5):
    for batch in train_loader:
        energy_optimizer.zero_grad()
        model_optimizer.zero_grad()
        images, labels = batch
        images = images.float()
        labels = labels.float()

        logits = vae.encode(images)
        probabilities = F.softmax(logits, dim=2)

        probabilities_condensed = probabilities.view(-1, num_logits)
        sampled_vector = sampler(probabilities_condensed, 1, True).view(
            batch_size, latent_vector_dim
        )

        valid_vector = scale_vector_copy_gradient(sampled_vector, probabilities)

        decoded = vae.decode(valid_vector)
        vae_loss = F.mse_loss(images, decoded)

        # energies = vae.calc_energy(valid_vector)
        energies = energyFn(valid_vector)

        energy_loss = energy_loss_fn(energies, labels)
        # norm = torch.norm(vae.coefficients)
        norm = calc_norm(energyFn.coefficients)
        norm_loss = F.mse_loss(norm, torch.ones_like(norm))

        energy_loss = energy_loss * 0.01
        vae_loss = vae_loss * 10
        norm_loss = norm_loss * 10

        total_loss = vae_loss + energy_loss + norm_loss

        total_loss.backward()

        energy_optimizer.step()
        model_optimizer.step()

        if i % 20 == 0:
            print("------------------------------")
            print(f"Energy loss: {energy_loss}")
            print(f"Model loss: {vae_loss}")
            print(f"Norm: {norm.item()}")
            print(f"Epoch: {epoch}")
            energy_loss_fn.print_info()
            plt.figure()
            plt.scatter(
                energies.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
            )
            plt.xlabel("energies")
            plt.ylabel("FOMs")
            plt.savefig("test_learnable_energyfn.png", dpi=300)

            og = (
                torchvision.utils.make_grid(expand_output(images, batch_size))
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            recon = (
                torchvision.utils.make_grid(
                    expand_output(mean_normalize(decoded), batch_size)
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )

            plt.imsave("original_images_test.png", og, dpi=300)
            plt.imsave("reconstruction_test.png", recon, dpi=300)

        i += 1
