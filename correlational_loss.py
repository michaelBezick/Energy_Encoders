import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from LDM_Classes import CorrelationalLoss

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

class Vector_Creator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ParameterList([nn.Linear(1, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 64)])
    def forward(self, x):
        x = self.layers(x)
        return x.view(-1, 64)

class Energy_Function(nn.Module):
    def __init__(self, dim, energy_function):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(dim, dim))
        self.weights = nn.init.xavier_uniform_(self.weights)
        self.energy_function = energy_function
    def forward(self, x):
        return self.energy_function(x, self.weights)


device = "cuda"

sample_vectors = torch.ones(2000, 64) * 0.5
sample_vectors = torch.bernoulli(sample_vectors)

FOM_labels = np.load("./FOM_labels.npy")

FOM_labels = torch.from_numpy(FOM_labels)

energyFunction = Energy_Function(64, quboEnergy).to(device)
loss_fn = CorrelationalLoss(correlation_weight=10, energy_weight=.01, slope_weight=0)
optimizer = torch.optim.Adam(params = energyFunction.parameters())

sample_vectors = sample_vectors.cuda()
FOM_labels = FOM_labels.cuda()

epochs = 10000
batch_size = 100
num_batches = 2000 // batch_size
for epoch in range(epochs):
    if epoch > 100 and epoch % 1000 == 0:
        x_list = torch.squeeze(FOM_labels).detach().cpu().numpy()
        y_list = torch.squeeze(energyFunction(sample_vectors)).detach().cpu().numpy()
        plt.scatter(x_list, y_list)
        plt.xlabel("FOM")
        plt.ylabel("Energy")
        plt.show()
    for i in range(num_batches):
        optimizer.zero_grad()
        vectors = sample_vectors[100 * i : 100 * (i + 1), :]
        FOMs = FOM_labels[100 * i : 100 * (i + 1), :]
        energies = energyFunction(vectors)
        energy_loss = loss_fn(FOMs, energies)
        if (i == 0 and epoch % 100 == 0):
            loss_fn.print_losses()
            loss_fn.print_info()
        energy_loss.backward()

        optimizer.step()
