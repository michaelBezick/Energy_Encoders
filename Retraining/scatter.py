import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


dataset = torch.load("./Correlational_Loss_Loss_fn_-1_degree_new_vector_labeled_dataset.pt")
print(dataset)
energy_fn = torch.load("./Correlational_Loss_Loss_fn_-1_newly_trained_energy_fn_weights.pt")
print(energy_fn)


dataloader = DataLoader(dataset, batch_size=100, drop_last=True)

FOMs_total = []

energies = []

with torch.no_grad():
    for vectors, FOMs in dataloader:
        FOMs_total.extend(FOMs.flatten().cpu().numpy())
        energies.extend(energy_fn(vectors).flatten().cpu().numpy())

corr = np.corrcoef(FOMs_total, energies)[0, 1]
print(corr)

plt.figure()
plt.scatter(FOMs_total, energies)
plt.xlabel("FOM")
plt.ylabel("Energies")
plt.savefig("scatter.pdf")

torch.save(torch.Tensor(FOMs_total), "FOMs_total.pt")
torch.save(torch.Tensor(energies), "Energies_total.pt")
