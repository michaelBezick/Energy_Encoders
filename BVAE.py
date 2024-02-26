import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from LDM_Classes import VAE_GAN, LabeledDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms as transforms
import tensorflow as tf
import os

num_MCMC_iterations = 0
temperature = 0.1
resume_from_checkpoint = True
min_energy_vector = -4.25

###############################################################

temperature_str = str(temperature).replace('.', ',')

experiment_name = f"{num_MCMC_iterations}_MCMC_temp_{temperature_str}"
if resume_from_checkpoint:
    checkpoint_path1 = f"./logs/{num_MCMC_iterations}_MCMC_temp_{temperature_str}/"
    checkpoint_path2 = os.listdir(checkpoint_path1)[-1]
    checkpoint_path = os.path.join(checkpoint_path1, checkpoint_path2)
    checkpoint_path = checkpoint_path + "/checkpoints/"
    file_checkpoint = os.listdir(checkpoint_path)[0]
    checkpoint_path = os.path.join(checkpoint_path, file_checkpoint)

num_devices = 3
num_nodes = 4
num_workers = 1
accelerator = "gpu"
batch_size = 100
epochs = -1

reconstruction_term = 0.6
perceptual_term = 0.025
kl_term = 0
adv_term = 0#1
energy_loss_term = 1.5
lr = 1e-3
latent_vector_dim=64 

checkpoint_callback = ModelCheckpoint(filename = "good", every_n_train_steps = 300)

torch.set_float32_matmul_precision('high')

QUBO_matrix = torch.load("QUBO_matrix.pt")
vae = VAE_GAN(reconstruction_term, perceptual_term, kl_term, adv_term, energy_loss_term, 1, 128, lr = lr, num_MCMC_iterations=num_MCMC_iterations, temperature = temperature, latent_vector_dim=latent_vector_dim, batch_size=batch_size, QUBO_matrix=QUBO_matrix, min_energy_vector = min_energy_vector) #previous dim was 128
dataset = np.expand_dims(np.load("top_0.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    
normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

labels = np.squeeze(np.load('FOM_labels.npy'))

labeled_dataset = LabeledDataset(dataset, labels)

train_loader = DataLoader(labeled_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = True, drop_last = True)

logger = TensorBoardLogger(save_dir='logs/', name=experiment_name)

lr_monitor = LearningRateMonitor(logging_interval = 'step')
trainer = pl.Trainer(logger = logger, devices = num_devices, num_nodes = num_nodes, accelerator = "gpu", detect_anomaly=True, log_every_n_steps=2, max_epochs=epochs)


if resume_from_checkpoint:
    trainer.fit(model = vae, train_dataloaders = train_loader, ckpt_path = checkpoint_path)
else:
    trainer.fit(model = vae, train_dataloaders = train_loader)
