import polytensor.polytensor as polytensor
import torch

num_MCMC_iterations = 1
temperature = 0.1
resume_from_checkpoint = False
num_devices = 2
num_nodes = 2
num_workers = 1
epochs = 10_000
reconstruction_weight = 0.6
perceptual_weight = 0.025
energy_weight = 1e-3
h_dim = 128
batch_size = 100
num_vars = 64

num_per_degree = [num_vars, num_vars * (num_vars - 1) // 2]
sample_fn = lambda: torch.randn(1, device='cuda')
terms = polytensor.generators.coeffPUBORandomSampler(
        n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
        )

energy_fn = polytensor.polynomial.PottsModel(terms)
torch.save(energy_fn, "Potts_energy_fn.pt")
