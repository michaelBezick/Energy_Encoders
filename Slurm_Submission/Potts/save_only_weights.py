import polytensor.polytensor as polytensor
import torch

model = torch.load("Potts_energy_fn.pt")
torch.save(model.interactions, "Potts_energy_fn_weights.pt")
