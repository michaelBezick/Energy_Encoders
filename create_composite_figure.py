import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import polytensor

from Energy_Encoder_Classes import BVAE, CorrelationalLoss, LabeledDataset
from Functions import (
    clamp_output,
)

from annealing_function_new import Annealing_Learnable_and_Matching

device = "cuda"
save_images = False

clamp = True
threshold = 0.5

num_vars = 64

num_logits = 2
num_per_degree = [num_vars]
sample_fn = lambda: torch.zeros(1, device="cuda")
terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

energy_loss_fn = CorrelationalLoss()

terms = polytensor.generators.denseFromSparse(terms, num_vars)
terms.append(torch.zeros(num_vars, num_vars))
terms.append(torch.zeros(num_vars, num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

# third_degree_model_correlation = BVAE.load_from_checkpoint("./Annealing_Learnable/Models/QUBO_order_3/epoch=9999-step=200000.ckpt", energy_fn=energy_fn, energy_loss_fn=energy_loss_fn, h_dim=128)

third_degree_model_matching = BVAE.load_from_checkpoint("./Annealing_Matching/Models/QUBO_order_3/epoch=9999-step=200000.ckpt", energy_fn=energy_fn, energy_loss_fn=energy_loss_fn, h_dim=128)

log_dir = ""

def mean_normalize(images: torch.Tensor):
    return (images - torch.min(images)) / (torch.max(images) - torch.min(images))

largest_FOM_global = 0

model_number = 0

# energies_correlation = []
# FOMs_correlation = []

energies_matching = []
FOMs_matching = []

dataset = np.expand_dims(np.load("./Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

dataset = clamp_output(dataset, 0.5)

labels = torch.load("./Files/FOM_labels_new.pt")

labeled_dataset = LabeledDataset(dataset, labels)

sampler = torch.multinomial

train_loader = DataLoader(labeled_dataset, num_workers = 1, batch_size = 100, shuffle = False, drop_last = True)

batch_size = 100
latent_vector_dim = 64
num_logits = 2

# third_degree_model_correlation = third_degree_model_correlation.to("cuda")
# third_degree_model_correlation.scale = third_degree_model_correlation.scale.cuda()

third_degree_model_matching = third_degree_model_matching.to("cuda")
third_degree_model_matching.scale = third_degree_model_matching.scale.cuda()

# """Correlation"""
# with torch.no_grad():
#     for batch in train_loader:
#         images, labels = batch
#         images = images.cuda().float()
#         labels = labels.cuda().float()
#
#         FOMs_correlation.extend(labels.detach().cpu().numpy())
#         logits = third_degree_model_correlation.vae.encode(images)
#         probabilities = F.softmax(logits, dim=2)
#         probabilities_condensed = probabilities.view(-1, num_logits)
#         sampled_vector = sampler(probabilities_condensed, 1, True).view(
#             batch_size, latent_vector_dim
#         )
#         valid_vector = third_degree_model_correlation.scale_vector_copy_gradient(sampled_vector, probabilities)
#         original_sampled_vector_with_gradient = valid_vector
#         sampled_energies = third_degree_model_correlation.energy_fn(original_sampled_vector_with_gradient)
#         sampled_energies = torch.squeeze(sampled_energies)
#         energies_correlation.extend(sampled_energies.detach().cpu().numpy())

# """TEST TEST TEST"""
# plt.scatter(FOMs_correlation, energies_correlation)
# plt.title("Third Degree Energies versus FOMs")
# plt.xlabel("FOMs")
# plt.ylabel("Energies")
# plt.savefig("TEST TEST TEST.png", dpi=300)
# exit()


"""Matching"""
with torch.no_grad():
    for batch in train_loader:
        images, labels = batch
        images = images.cuda().float()
        labels = labels.cuda().float()

        FOMs_matching.extend(labels.detach().cpu().numpy())
        logits = third_degree_model_matching.vae.encode(images)
        probabilities = F.softmax(logits, dim=2)
        probabilities_condensed = probabilities.view(-1, num_logits)
        sampled_vector = sampler(probabilities_condensed, 1, True).view(
            batch_size, latent_vector_dim
        )
        valid_vector = third_degree_model_matching.scale_vector_copy_gradient(sampled_vector, probabilities)
        original_sampled_vector_with_gradient = valid_vector
        sampled_energies = third_degree_model_matching.energy_fn(original_sampled_vector_with_gradient)
        sampled_energies = torch.squeeze(sampled_energies)
        energies_matching.extend(sampled_energies.detach().cpu().numpy())

"""NOW NEED TO GET ANNEALING PROGRESS FOR BOTH"""

"""CHANGED"""
[annealing_energies_matching] =  Annealing_Learnable_and_Matching()


# plt.figure()
# plt.scatter(FOMs_correlation, energies_correlation, label="Learned Correlation on Dataset")
# plt.xlabel("FOM")
# plt.ylabel("Energy")
# plt.title("Pearson Correlation Annealing Overlay")
# scaled_list_correlation = np.linspace(0.2, 1.2, len(annealing_energies_correlation))
# plt.plot(scaled_list_correlation, annealing_energies_correlation, color='red', label="VNA Average Energy Solution Curve")
# plt.legend()
# plt.savefig("Pearson_Correlation_with_Annealing_Overlay.png", dpi=500)

plt.figure()
plt.scatter(FOMs_matching, energies_matching, label="Learned Correlation on Dataset")
plt.xlabel("FOM")
plt.ylabel("Energy")
plt.title("Energy Matching with Annealing Overlay")
scaled_list_matching = np.linspace(0.2, 1.2, len(annealing_energies_matching))
plt.plot(scaled_list_matching, annealing_energies_matching, color='red', label="VNA Average Energy Solution Curve")
plt.legend()
plt.savefig("Energy_Matching_with_Annealing_Overlay.png", dpi=500)
