import torch
import torch.nn as nn
import torch.nn.functional as F
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

class Energy_Function(nn.Module):
    def __init__(self, dim, energy_function):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(dim, dim))
        self.weights = nn.init.xavier_uniform_(self.weights)
        self.energy_function = energy_function
    def forward(self, x):
        return self.energy_function(x, self.weights / torch.norm(self.weights))

class Vector_Creator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.layers(x)
        x = self.sigmoid(x)
        return x.view(-1, 64)


device = "cuda"

QUBO_matrix = torch.load("./QUBO_matrix.pt").cuda()

FOM_labels = np.load("./FOM_labels.npy")

FOM_labels = torch.from_numpy(FOM_labels)

loss_fn = CorrelationalLoss(correlation_weight=1, energy_weight=1, slope_weight=0)
vc = Vector_Creator().to(device)
optimizer_vc = torch.optim.Adam(params = vc.parameters(), lr=1e-3)
FOM_labels = FOM_labels.cuda()

epochs = 2000
batch_size = 100
num_batches = 2000 // batch_size

energy_list = []
losses = []
for epoch in range(epochs):
    if epoch > 100 and epoch % 100 == 0:
        x_list = torch.squeeze(FOM_labels).detach().cpu().numpy()
        y_list = energy_list
        plt.scatter(x_list, y_list)
        plt.xlabel("FOM")
        plt.ylabel("Energy")
        plt.savefig("correlation.png")



    energy_list = []
    for i in range(num_batches):
        optimizer_vc.zero_grad()
        FOMs = FOM_labels[100 * i : 100 * (i + 1), :]
        vectors = vc(FOMs)
        vectors_sampled = torch.bernoulli(vectors)
        vectors = (vectors_sampled - vectors.detach()) + vectors
        energies = quboEnergy(vectors, QUBO_matrix)

        energies_np = energies.detach().cpu().numpy()
        for energy_np in energies_np:
            energy_list.append(energy_np)


        energy_loss = loss_fn(FOMs, energies)

        losses.append(energy_loss.detach().cpu().item())
        if (i == 0 and epoch % 100 == 0):
            loss_fn.print_losses()
            loss_fn.print_info()
        energy_loss.backward()

        optimizer_vc.step()


plt.clf()
plt.plot(np.linspace(0, epochs, len(losses)), losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("losses.png")
