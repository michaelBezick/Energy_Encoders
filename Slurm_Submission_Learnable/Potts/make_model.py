import torch
from Energy_Encoder_Modules import Potts_Energy_Fn

interaction_terms = torch.randn((64, 64))

energy_fn = Potts_Energy_Fn(interaction_terms)

torch.save(energy_fn, "Potts_energy_fn.pt")
