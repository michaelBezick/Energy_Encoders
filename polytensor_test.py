import polytensor
import torch

num_vars = 64
sample_fn = lambda: torch.randn(1)
num_per_degree = [64, int(64 * 63 / 2)]

terms = polytensor.generators.coeffPUBORandomSampler(
            n=num_vars, num_terms=num_per_degree, sample_fn = sample_fn
        )

energy_fn = polytensor.SparsePolynomial(terms, "cuda")

vector = torch.bernoulli(0.5 * torch.ones(64)).float()
polytensor.serialize.tojson(energy_fn)
