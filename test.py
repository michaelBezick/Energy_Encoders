import torch
from torch.autograd import functional
from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn
from Energy_Encoder_Modules import Potts_Energy_Fn


#blume capel example

scale = torch.Tensor([-1, 0, 1])
logits = torch.randn(100, 6, 3)
logits = F.softmax(logits, dim=2)
logits = logits.view(-1, 3)
sampled = torch.multinomial(logits, 1, True)
sampled = sampled.view(100, 6)
one_hot = F.one_hot(sampled)
logits = logits.view(100, 6, 3)
copied_grad = (one_hot - logits).detach() + logits
print(copied_grad)
print(copied_grad.size())
scaled = copied_grad * scale
print(scaled.size())
summed = torch.einsum("ijk->ij", scaled)
print(summed[0, :])
print(one_hot[0, :, :])
exit()





interactions = torch.randn(64, 64)
interactions = torch.triu(interactions)
vector = 0.5 * torch.ones(64)
vector = torch.bernoulli(vector)
energy = Potts_Energy_Fn(vector, interactions)
print(energy)
exit()

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

interactions = torch.randn(64, 64)
interactions = torch.triu(interactions)

vector_logits = torch.randn(64, 3)
vector_probabilities = F.softmax(vector_logits)
print(vector_probabilities)
exit()
sampled = torch.multinomial(vector_probabilities, 1, True)
scaled = sampled - 1.
print(scaled.size())
print(interactions.size())
energy = torch.einsum("kj,ji,ik->k", scaled.t(), interactions, scaled)
exit()


class TensorizedLayer(nn.Module):
    def __init__(self, dim_hidden, dim_sigma):
        super().__init__()
        self.W = nn.Parameter(torch.randn((dim_hidden, dim_sigma, dim_hidden)))
        self.b = nn.Parameter(torch.randn((dim_hidden)))

    def forward(self, sigma, hidden):
        x = torch.einsum("ij,kjl->kl", sigma.t(), self.W)
        x = torch.einsum("ij,jk->ik", x, hidden)
        x = x + self.b
        return x

logits = torch.randn(10, 2)

probabilities = F.softmax(logits)
print(probabilities)
sampler = torch.multinomial
sampled = sampler(probabilities, 1, replacement=True)
print(sampled)
print(sampled.size())
#print(f"Num -1: {torch.sum(sampled.eq(-1)).item()}")
#print(f"Num 0: {torch.sum(sampled.eq(0)).item()}")
#print(f"Num 1: {torch.sum(sampled.eq(1)).item()}")

exit()
layer = TensorizedLayer(dim_hidden=4, dim_sigma=2)
sigma = torch.Tensor([[1],
                      [0]])
hidden = torch.Tensor([[1],
                       [2],
                       [3],
                       [4]])
out = layer(sigma, hidden)
print(out)

exit()
k = 2
n = 3
v1 = torch.Tensor([[1, 2]])
v2 = torch.Tensor([[[3, 4, 5],
                    [6, 7, 8]],
                   [[9, 10, 11],
                    [100, 200, 300]]
                   ])
v4 = torch.Tensor([1, 2, 3])
v1 = torch.randn(1, k)
v2 = torch.randn(k, k, n)
v4 = torch.randn(n)
#v1 needs to be a row vector
#(1xk)(kxkxn)(nx1) = (kx1)
v3 = torch.einsum("ij,kjl->kl", v1, v2)
v5 = torch.einsum("kl,l->k", v3, v4)
