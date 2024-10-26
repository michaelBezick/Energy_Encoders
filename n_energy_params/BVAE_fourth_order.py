import os

import numpy as np
import polytensor.polytensor as polytensor
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from Energy_Encoder_Classes import BVAE, CorrelationalLoss, LabeledDataset, Model_Type
from Energy_Encoder_Modules import calc_norm

num_MCMC_iterations = 0
temperature = 0.1
resume_from_checkpoint = True
num_devices = 3
num_nodes = 4
num_workers = 1
epochs = 10000
reconstruction_weight = 0.6
perceptual_weight = 0.025
energy_weight = 1e-3
norm_weight = 10
h_dim = 128
batch_size = 100
num_vars = 64
model_type = Model_Type.QUBO
order = 4

###############################################################


num_vars = 64
#terms to distribute: 2080

num_per_degree = [
    num_vars, #max 64
    num_vars * (num_vars - 1) // 2, #max 2016
    num_vars * (num_vars - 1) * (num_vars - 2) // 6, #max 41,664
    num_vars * (num_vars - 1) * (num_vars - 2) * (num_vars - 3) // 24, #max 635,376
]

num_per_degree = [
    64, #max 64
    2016, #max 2016
    10_000, #max 41,664
    20_000, #max 635,376
]

num_per_degree = [num_vars]

sample_fn = lambda: torch.randn(1, device="cuda")

terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

terms = polytensor.generators.denseFromSparse(terms, num_vars)
terms.append(torch.randn(num_vars, num_vars))
terms.append(torch.randn(num_vars, num_vars, num_vars))
terms.append(torch.randn(num_vars, num_vars, num_vars, num_vars))

norm = calc_norm(terms)
print(norm)
terms[0] = terms[0] / norm
terms[1] = terms[1] / norm
terms[2] = terms[2] / norm
terms[3] = terms[3] / norm
norm = calc_norm(terms)
print(norm)

energy_fn = polytensor.DensePolynomial(terms)

correlation_weight=10.0
energy_weight=.01
slope_weight=0.0

energy_loss_fn = CorrelationalLoss(correlation_weight, energy_weight, slope_weight)

bvae = BVAE(
    energy_fn,
    energy_loss_fn,
    model_type=model_type,
    reconstruction_weight=reconstruction_weight,
    perceptual_weight=perceptual_weight,
    energy_weight=energy_weight,
    norm_weight=norm_weight,
    h_dim=h_dim,
    latent_vector_dim=num_vars,
    num_MCMC_iterations=num_MCMC_iterations,
    temperature=temperature,
    batch_size=batch_size,
)

temperature_str = str(temperature).replace(".", ",")
model_type_str = bvae.model_type

experiment_name = f"{model_type_str}_order_{order}_NEW_HYPERPARAMETER"

checkpoint_path = ""
if resume_from_checkpoint:
    checkpoint_path1 = (
        f"./logs/{model_type_str}_order_{order}_NEW_HYPERPARAMETER/"
    )
    checkpoint_path2 = os.listdir(checkpoint_path1)
    newest_version = ""
    version_num_max = -1
    for version in checkpoint_path2:
        version_num = int(version.split("_")[1])
        if version_num > version_num_max:
            version_num_max = version_num
            newest_version = version
    checkpoint_path = os.path.join(checkpoint_path1, newest_version)
    checkpoint_path = checkpoint_path + "/checkpoints/"
    file_checkpoint = os.listdir(checkpoint_path)[0]
    checkpoint_path = os.path.join(checkpoint_path, file_checkpoint)
    print(f"Checkpoint path: {checkpoint_path}")

checkpoint_callback = ModelCheckpoint(filename="good", every_n_train_steps=300)

torch.set_float32_matmul_precision("high")


def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))


dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

dataset = clamp_output(dataset, 0.5)

labels = torch.load("../Files/FOM_labels_new.pt")

labeled_dataset = LabeledDataset(dataset, labels)

train_loader = DataLoader(
    labeled_dataset,
    num_workers=num_workers,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

logger = TensorBoardLogger(save_dir="logs/", name=experiment_name)

lr_monitor = LearningRateMonitor(logging_interval="step")
trainer = pl.Trainer(
    logger=logger,
    devices=num_devices,
    num_nodes=num_nodes,
    accelerator="gpu",
    log_every_n_steps=2,
    max_epochs=epochs,
)

if resume_from_checkpoint:
    trainer.fit(model=bvae, train_dataloaders=train_loader, ckpt_path=checkpoint_path)
else:
    trainer.fit(model=bvae, train_dataloaders=train_loader)
