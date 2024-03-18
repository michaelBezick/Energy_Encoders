import polytensor.polytensor as polytensor
import torch

model = torch.load("QUBO_energy_fn.pt")
torch.save(model.coefficients, "QUBO_energy_fn_coefficients.pt")
