from torch import nn
import math as m
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import Dataset
from Energy_Encoder_Modules import AttnBlock, VGGPerceptualLoss, ResnetBlockVAE
from enum import Enum

class Model_Type(Enum):
    QUBO = 1
    PUBO = 2
    ISING = 3
    BLUME_CAPEL = 4
    POTTS = 5

def blume_capel_scale(x):
    return x - 1
def ising_scale(x):
    return 2 * x - 1
def no_scale(x):
    return x

class BVAE(pl.LightningModule):
    def __init__(self, energy_fn, energy_loss_fn, model_type = Model_Type.QUBO, reconstruction_weight=0., perceptual_weight=0., energy_weight=0., in_channels = 1, h_dim = 32, lr = 1e-3, batch_size = 100, num_MCMC_iterations = 3, temperature = 0.1, latent_vector_dim = 64):
        super().__init__()

        self.sampler = torch.multinomial
        if model_type == Model_Type.QUBO:
            self.model_type = 'QUBO'
            num_logits = 2
            self.scale = torch.Tensor([0., 1.])
            self.shift = 0.
        elif model_type == Model_Type.PUBO:
            self.model_type = 'PUBO'
            num_logits = 2
            self.scale = torch.Tensor([0., 1.])
            self.shift = 0.
        elif model_type == Model_Type.BLUME_CAPEL:
            self.model_type = 'Blume-Capel'
            num_logits = 3
            self.scale = torch.Tensor([-1., 0., 1.])
            self.shift = 1.
        elif model_type == Model_Type.POTTS:
            self.model_type = 'Potts'
            num_logits = 2
            self.scale = torch.Tensor([0., 1.])
            self.shift = 0
        else:
            raise ValueError("Model does not exist!")

        self.automatic_optimization = False
        self.num_logits = num_logits

        self.batch_size = batch_size
        self.latent_vector_dim = latent_vector_dim

        self.lr = lr
        self.vae = VAE(in_channels, h_dim, lr, batch_size, num_logits, latent_vector_dim)

        self.num_MCMC_iterations = num_MCMC_iterations

        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.energy_weight = energy_weight

        self.temperature = temperature

        self.energy_fn = energy_fn
        self.energy_loss_fn = energy_loss_fn

        self.perceptual_loss = VGGPerceptualLoss()
        self.perceptual_loss.eval()

        self.sigmoid = nn.Sigmoid()

    def MCMC_step(self, initial_vectors):
        #get random indices
        indices = torch.randint(0, self.latent_vector_dim, (self.batch_size, 1), device=self.device)

        #get new spins corresponding to indices
        new_spins = torch.randint(0, self.num_logits, (self.batch_size, 1), device=self.device)
        new_spins = F.one_hot(new_spins, self.num_logits).float()
        new_spins = torch.einsum("ijk,k->ij", new_spins, self.scale)

        transitioned_vectors = initial_vectors.scatter(dim=1, index=indices, src=new_spins)

        #need to move to transitioned vectors with acceptance probabilities
        initial_energy = torch.squeeze(self.energy_fn(initial_vectors))
        transitioned_energy = torch.squeeze(self.energy_fn(transitioned_vectors))
        e_matrix = m.e * torch.ones((self.batch_size), device=self.device)
        acceptance_prob_RHS = torch.pow(e_matrix, (initial_energy - transitioned_energy) / (self.temperature))        
        acceptance_prob_LHS = torch.ones((self.batch_size), device=self.device)
        acceptance_probability = torch.min(acceptance_prob_LHS, acceptance_prob_RHS)

        acceptance_sample = torch.bernoulli(acceptance_probability).unsqueeze(1).int()
        acceptance_sample_expanded = acceptance_sample.expand(self.batch_size, self.latent_vector_dim)

        #if acceptance_sample = 1, move to next one
        output = torch.where(acceptance_sample_expanded == 1, transitioned_vectors, initial_vectors)

        return output

    def scale_vector_copy_gradient(self, x, probabilities):
        '''
        x in index format -> x in scaled format with gradient
        '''
        x = F.one_hot(x)
        copied_grad = (x - probabilities).detach() + probabilities
        return torch.einsum("ijk,k->ij", copied_grad, self.scale)

    def training_step(self, batch, batch_idx):
        x, FOM_labels = batch

        opt_VAE= self.optimizers()
        scheduler = self.lr_schedulers()

        #training VAE
        self.toggle_optimizer(opt_VAE)

        logits = self.vae.encode(x)

        probabilities = F.softmax(logits, dim=2)

        probabilities_condensed = probabilities.view(-1, self.num_logits)
        sampled_vector = self.sampler(probabilities_condensed, 1, True).view(self.batch_size, self.latent_vector_dim)
        
        valid_vector = self.scale_vector_copy_gradient(sampled_vector, probabilities)

        """"""
        original_sampled_vector_with_gradient = valid_vector.clone()
        """"""

        transitioned_vectors = valid_vector.detach()
        for _ in range(self.num_MCMC_iterations):
            transitioned_vectors = self.MCMC_step(transitioned_vectors)

        shifted_vector = transitioned_vectors + self.shift

        """"""
        transitioned_vectors_with_gradient = self.scale_vector_copy_gradient(shifted_vector.long(), probabilities)
        """"""

        x_hat = self.vae.decode(transitioned_vectors_with_gradient)

        #logging generated images
        sample_imgs_generated = x_hat[:30]
        sample_imgs_original = x[:30]
        gridGenerated = torchvision.utils.make_grid(sample_imgs_generated)
        gridOriginal = torchvision.utils.make_grid(sample_imgs_original)

        """Tensorboard logging"""

        if self.global_step % 1000 == 0:
            self.logger.experiment.add_image("Generated_images", gridGenerated, self.global_step)
            self.logger.experiment.add_image("Original_images", gridOriginal, self.global_step)

        #reconstruction quality
        reconstruction_loss = F.mse_loss(x_hat, x) * self.reconstruction_weight
        perceptual_loss_value = self.perceptual_loss(x_hat, x) * self.perceptual_weight

        #energy correlation
        print(self.energy_fn)
        #print(self.energy_fn.coefficients)
        #print(self.energy_fn.terms)
        print(original_sampled_vector_with_gradient)
        energy = self.energy_fn(original_sampled_vector_with_gradient)
        energy_loss = self.energy_loss_fn(FOM_labels, energy) * self.energy_weight

        total_loss = perceptual_loss_value + reconstruction_loss + energy_loss
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("perceptual_loss", perceptual_loss_value)
        self.log("energy_loss", energy_loss, prog_bar=True, on_step=True)
        self.log("pearson_correlation_coefficient", self.energy_loss_fn.correlation)
        self.log("train_loss", total_loss, prog_bar=True, on_step=True)

        self.manual_backward(total_loss)
        opt_VAE.step()
        scheduler.step()
        opt_VAE.zero_grad()
        self.untoggle_optimizer(opt_VAE)

    def configure_optimizers(self):
        opt_VAE = torch.optim.Adam(self.vae.parameters(), lr = self.lr) #weight_decay=0.01
        scheduler = optim.lr_scheduler.LambdaLR(optimizer = opt_VAE, lr_lambda = lambda epoch: self.warmup_lr_schedule(epoch))

        return ({"optimizer": opt_VAE, "lr_scheduler": scheduler}
                 )
    
    def warmup_lr_schedule(self, epoch):
        warmup_epochs = 400
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1
    def on_train_start(self):
        self.energy_fn = self.energy_fn.to(self.device)
        self.scale = self.scale.to(self.device)

class CorrelationalLoss():
    """
    Idea:
    1. want the correlation to be closest to -1
    2. want the average energy to be minimized
    3. want the slope of the linear correlation to be minimized
    """
    def __init__(self, correlation_weight = 1., energy_weight = 1., slope_weight = 1.):

        self.correlation_weight = correlation_weight
        self.energy_weight = energy_weight
        self.slope_weight = slope_weight

        self.covariance = 0
        self.std_dev_x = 0

        self.correlation_loss = 0
        self.average_energy_loss = 0
        self.slope_loss = 0

        self.correlation = 0
        self.average_energy = 0
        self.slope = 0

    def __call__(self, x_FOM, y_Energy):
        pearson_correlation_coefficient = self.compute_pearson_correlation(x_FOM, y_Energy)
        average_energy = self.compute_average_energy(y_Energy)
        slope = self.compute_slope()

        self.correlation = pearson_correlation_coefficient
        self.average_energy = average_energy
        self.slope = slope

        x = pearson_correlation_coefficient
        correlation_loss = torch.log((0.5 * (x + 1))/(1 - 0.5 * (x + 1)))

        average_energy_loss = average_energy

        slope_loss = slope

        loss_combined = self.correlation_weight * correlation_loss + \
                        self.energy_weight * average_energy_loss + \
                        self.slope_weight * slope_loss
        
        self.correlation_loss = correlation_loss * self.correlation_weight
        self.average_energy_loss = average_energy_loss * self.energy_weight
        self.slope_loss = slope_loss * self.slope_weight

        return loss_combined


    def print_info(self):
        print(f"correlation: {self.correlation}\taverage_energy: {self.average_energy}\tslope: {self.slope}")

    def print_losses(self):
        print(f"correlation_loss: {self.correlation_loss}\tenergy_loss: {self.average_energy_loss}\tslope_loss: {self.slope_loss}")
    def compute_slope(self):
        return self.covariance / self.std_dev_x

    def compute_average_energy(self, y_Energy):
        return torch.mean(y_Energy)

    def compute_pearson_correlation(self, x_FOM, y_Energy):

        #x should be a vector of length n
        #y should be a vector of length n

        x_FOM = torch.squeeze(x_FOM)
        y_Energy = torch.squeeze(y_Energy)
        x_mean = torch.mean(x_FOM)
        y_mean = torch.mean(y_Energy)

        x_deviation_from_mean = x_FOM - x_mean
        y_deviation_from_mean = y_Energy - y_mean

        covariance = torch.einsum("i,i->", x_deviation_from_mean, y_deviation_from_mean)

        if (covariance == 0):
            print("COVARIANCE 0")
            exit()

        std_dev_x = torch.sqrt(torch.einsum("i,i->", x_deviation_from_mean, x_deviation_from_mean))
        if (std_dev_x == 0):
            print("std_dev_x 0")
            exit()
        std_dev_y = torch.sqrt(torch.einsum("i,i->", y_deviation_from_mean, y_deviation_from_mean))
        if (std_dev_y == 0):
            print("std_dev_y 0")
            exit()

        pearson_correlation_coefficient = covariance / (std_dev_x * std_dev_y)

        self.covariance = covariance
        self.std_dev_x = std_dev_x

        return pearson_correlation_coefficient

class LabeledDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image[:, 0:32, 0:32], label

class VAE(pl.LightningModule):
    '''
    Variational autoencoder with UNet structure.
    '''
    def __init__(self, in_channels = 1, h_dim = 32, lr = 1e-3, batch_size = 100, num_logits = 1, problem_size = 64):
        super().__init__()

        self.batch_size = batch_size

        self.lr = lr
        self.problem_size = problem_size
        self.num_logits = num_logits

        self.attention1E = AttnBlock(h_dim)
        self.attention2E = AttnBlock(h_dim)
        self.resnet1E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet2E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet3E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet4E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet5E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet6E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.maxPool = nn.MaxPool2d((2, 2), 2)

        self.encoder = nn.Sequential(nn.Conv2d(in_channels, h_dim, kernel_size = (3, 3), padding = 'same'),
                                    nn.SiLU(),
                                    self.resnet1E,
                                    self.resnet2E,
                                    self.maxPool, # 16 x 16
                                    self.resnet3E,
                                    self.attention1E,
                                    self.resnet4E,
                                    self.maxPool, # 8 x 8
                                    self.resnet5E,
                                    self.attention2E,
                                    self.resnet6E,
                                    nn.Conv2d(h_dim, 1, kernel_size=1),
                                    nn.Flatten(),
                                    nn.ReLU(),
                                    nn.Linear(64, problem_size * num_logits),
                                    )
        

        self.attention1D = AttnBlock(h_dim)
        self.attention2D = AttnBlock(h_dim)
        self.resnet1D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet2D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet3D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet4D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet5D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet6D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        
        self.decoder = nn.Sequential(nn.Linear(problem_size, 64),
                                     nn.ReLU(),
                                     nn.Unflatten(1, (1, 8, 8)),
                                     nn.Conv2d(1, h_dim, (1, 1)),
                                     self.resnet1D,
                                     self.attention1D,
                                     self.resnet2D,
                                     nn.ConvTranspose2d(h_dim, h_dim, (2, 2), 2), # 16 x 16
                                     self.resnet3D,
                                     self.attention2D,
                                     self.resnet4D,
                                     nn.ConvTranspose2d(h_dim, h_dim, (2, 2), 2), # 32 x 32
                                     self.resnet5D,
                                     self.resnet6D,
                                     nn.Conv2d(h_dim, 1, (1, 1))
                                     )
    def encode(self, x):
        logits = self.encoder(x)
        return logits.view(self.batch_size, self.problem_size, self.num_logits)
    
    def decode(self, z):
        return self.decoder(z)
