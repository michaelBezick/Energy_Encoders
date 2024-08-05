import numpy as np
import torchvision
import torch
from Energy_Encoder_Classes import BVAE, CorrelationalLoss, LabeledDataset
import polytensor as polytensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F

def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

def expand_output(tensor: torch.Tensor):
    x = torch.zeros([100, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x
device = "cuda"

threshold = 0.5

lowest_num = 5000

num_vars = 64

num_per_degree = [num_vars]
sample_fn = lambda: torch.randn(1, device="cuda")
terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

energy_loss_fn = CorrelationalLoss()

terms = polytensor.generators.denseFromSparse(terms, num_vars)
terms.append(torch.randn(num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

second_degree_model = BVAE.load_from_checkpoint("../Annealing_Learnable/Models/QUBO_order_2/epoch=9999-step=200000.ckpt", energy_fn=energy_fn, energy_loss_fn=energy_loss_fn, h_dim=128)

dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

dataset = clamp_output(dataset, 0.5)

labels = torch.load("../Files/FOM_labels_new.pt")

labeled_dataset = LabeledDataset(dataset, labels)
batch_size = 100
num_workers=1

train_loader = DataLoader(labeled_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = True, drop_last = True)

sampler = torch.multinomial

for batch in train_loader:
    images, labels = batch
    images = images.cuda()
    labels = labels.cuda()
    logits = second_degree_model.vae.encode(images)
    probabilities = F.softmax(logits, dim=2)
    probabilities_condensed = probabilities.view(-1, 2)
    sampled_vector = sampler(probabilities_condensed, 1, True).view(
        100,64
    )

    decoded = second_degree_model.vae.decode(sampled_vector.float())
    decoded = expand_output(decoded)
    decoded = clamp_output(decoded, 0.5)

    decoded_grid = torchvision.utils.make_grid(decoded.cpu()[0:16], nrow=4)

    images_grid = torchvision.utils.make_grid(expand_output(images.cpu())[0:16], nrow=4)

    save_image(decoded_grid, "decoded.png", dpi=300)
    save_image(images_grid, "original.png", dpi=300)
    exit()
