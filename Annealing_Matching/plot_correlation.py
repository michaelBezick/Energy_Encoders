import torch
import torch.nn.functional as F
import numpy as np
import polytensor
from Energy_Encoder_Classes import CorrelationalLoss
from Energy_Encoder_Classes import BVAE, LabeledDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

num_vars = 64

num_per_degree = [num_vars]
sample_fn = lambda: torch.randn(1, device="cuda")
terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

energy_loss_fn = CorrelationalLoss(1, 1, 1)

terms = polytensor.generators.denseFromSparse(terms, num_vars)
terms.append(torch.randn(num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)


second_degree_model = BVAE.load_from_checkpoint("./Models/QUBO_order_2/epoch=9999-step=200000.ckpt", energy_fn=energy_fn, energy_loss_fn=energy_loss_fn, h_dim=128)
terms.append(torch.randn(num_vars, num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

third_degree_model = BVAE.load_from_checkpoint("./Models/QUBO_order_3/epoch=9999-step=200000.ckpt", energy_fn=energy_fn, energy_loss_fn=energy_loss_fn, h_dim=128)

dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

dataset = clamp_output(dataset, 0.5)

labels = torch.load("../Files/FOM_labels_new.pt")

labeled_dataset = LabeledDataset(dataset, labels)

num_workers = 1
batch_size = 100
latent_vector_dim = 64
num_logits = 2
sampler = torch.multinomial
cl = CorrelationalLoss(1, 1, 1)

train_loader = DataLoader(labeled_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = True, drop_last = True)

"""Second Degree Model"""
second_degree_model = second_degree_model.to("cuda")

second_degree_model.scale = second_degree_model.scale.cuda()

FOMs = []
energies = []

with torch.no_grad():
    for batch in train_loader:
        images, labels = batch
        images = images.cuda().float()
        labels = labels.cuda().float()

        FOMs.extend(labels.detach().cpu().numpy())
        logits = second_degree_model.vae.encode(images)
        probabilities = F.softmax(logits, dim=2)
        probabilities_condensed = probabilities.view(-1, num_logits)
        sampled_vector = sampler(probabilities_condensed, 1, True).view(
            batch_size, latent_vector_dim
        )
        valid_vector = second_degree_model.scale_vector_copy_gradient(sampled_vector, probabilities)
        original_sampled_vector_with_gradient = valid_vector
        sampled_energies = second_degree_model.energy_fn(original_sampled_vector_with_gradient)
        sampled_energies = torch.squeeze(sampled_energies)
        loss = cl(sampled_energies, labels)
        energies.extend(sampled_energies.detach().cpu().numpy())

loss = cl(torch.Tensor(FOMs), torch.Tensor(energies))
second_degree_pearson = cl.correlation
plt.figure()
plt.scatter(FOMs, energies)
plt.title("Second Degree Energies versus FOMs")
plt.xlabel("FOMs")
plt.ylabel("Energies")
plt.savefig("Scatter_Second_Degree_Matching.png", dpi=300)

third_degree_model = third_degree_model.to("cuda")

third_degree_model.scale = third_degree_model.scale.cuda()

FOMs = []
energies = []

with torch.no_grad():
    for batch in train_loader:
        images, labels = batch
        images = images.cuda().float()
        labels = labels.cuda().float()

        FOMs.extend(labels.detach().cpu().numpy())
        logits = third_degree_model.vae.encode(images)
        probabilities = F.softmax(logits, dim=2)
        probabilities_condensed = probabilities.view(-1, num_logits)
        sampled_vector = sampler(probabilities_condensed, 1, True).view(
            batch_size, latent_vector_dim
        )
        valid_vector = third_degree_model.scale_vector_copy_gradient(sampled_vector, probabilities)
        original_sampled_vector_with_gradient = valid_vector
        sampled_energies = third_degree_model.energy_fn(original_sampled_vector_with_gradient)
        sampled_energies = torch.squeeze(sampled_energies)
        loss = cl(sampled_energies, labels)
        energies.extend(sampled_energies.detach().cpu().numpy())

loss = cl(torch.Tensor(FOMs), torch.Tensor(energies))
third_degree_pearson = cl.correlation
plt.figure()
plt.scatter(FOMs, energies)
plt.title("Third Degree Energies versus FOMs")
plt.xlabel("FOMs")
plt.ylabel("Energies")
plt.savefig("Scatter_Third_Degree_Matching.png", dpi=300)

with open("plot_notes.txt", "w") as file:
    file.write(f"Second_degree_pearson: {second_degree_pearson}\nThird_degree_pearson: {third_degree_pearson}")
