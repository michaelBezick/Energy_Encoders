import os
import numpy as np
import torch
from Modules.Energy_Encoder_Classes import BVAE, CorrelationalLoss, Model_Type, LabeledDataset
import polytensor.polytensor as polytensor
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl

num_MCMC_iterations = 0
temperature = 0.1
resume_from_checkpoint = False
num_devices = 2
num_nodes = 2
num_workers = 1
epochs = 10_000
reconstruction_weight = 0.6
perceptual_weight = 0.025
energy_weight = 1e-3
h_dim = 128
batch_size = 100
num_vars = 64
model_type = Model_Type.POTTS

###############################################################

energy_fn = torch.load("./Potts_energy_fn.pt")

energy_loss_fn = CorrelationalLoss(10., 0.01, 0.)

bvae = BVAE(energy_fn, energy_loss_fn, model_type=model_type, reconstruction_weight=reconstruction_weight, perceptual_weight=perceptual_weight, energy_weight=energy_weight, h_dim=h_dim, latent_vector_dim=num_vars, num_MCMC_iterations=num_MCMC_iterations, temperature=temperature, batch_size=batch_size)

temperature_str = str(temperature).replace('.', ',')
model_type_str = bvae.model_type

experiment_name = f"{model_type_str}_{num_MCMC_iterations}_MCMC_temp_{temperature_str}"

checkpoint_path = ""
if resume_from_checkpoint:
    checkpoint_path1 = f"./logs/{model_type_str}_{num_MCMC_iterations}_MCMC_temp_{temperature_str}/"
    checkpoint_path2 = os.listdir(checkpoint_path1)[-1]
    checkpoint_path = os.path.join(checkpoint_path1, checkpoint_path2)
    checkpoint_path = checkpoint_path + "/checkpoints/"
    file_checkpoint = os.listdir(checkpoint_path)[0]
    checkpoint_path = os.path.join(checkpoint_path, file_checkpoint)

checkpoint_callback = ModelCheckpoint(filename = "good", every_n_train_steps = 300)

torch.set_float32_matmul_precision('high')

dataset = np.expand_dims(np.load("top_0.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    
normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

labels = torch.from_numpy(np.squeeze(np.load('FOM_labels.npy')))

labeled_dataset = LabeledDataset(dataset, labels)

train_loader = DataLoader(labeled_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = True, drop_last = True)

#logger = CSVLogger(save_dir="logs/", name=experiment_name)
logger = TensorBoardLogger(save_dir="logs/", name=experiment_name)

lr_monitor = LearningRateMonitor(logging_interval = 'step')
trainer = pl.Trainer(logger = logger, devices = num_devices, num_nodes = num_nodes, accelerator = "gpu", log_every_n_steps=2, max_epochs=epochs)

if resume_from_checkpoint:
    trainer.fit(model = bvae, train_dataloaders = train_loader, ckpt_path = checkpoint_path)
else:
    trainer.fit(model = bvae, train_dataloaders = train_loader)
