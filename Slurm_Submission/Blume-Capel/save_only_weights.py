import polytensor.polytensor as polytensor
import torch

model = torch.load("Blume-Capel_energy_fn.pt")
torch.save(model.coefficients, "Blume-Capel_energy_fn_coefficients.pt")
