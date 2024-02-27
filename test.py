import torch
from torch.autograd import functional
from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn
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

logits = torch.Tensor([1, 1, 1])

probabilities = F.softmax(logits)
print(probabilities)
sampler = torch.multinomial
sampled = sampler(probabilities, 1, replacement=True)
sampled -= 1
print(sampled)
print(f"Num -1: {torch.sum(sampled.eq(-1)).item()}")
print(f"Num 0: {torch.sum(sampled.eq(0)).item()}")
print(f"Num 1: {torch.sum(sampled.eq(1)).item()}")

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
