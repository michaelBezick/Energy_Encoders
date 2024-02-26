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

#class TransformerEncoder(nn.Module):
#    def __init__(self, dim):
#        super().__init__()
#        self.FF = nn.Sequential(nn.Linear(dim, dim),
#                                nn.ReLU(),
#                                nn.Linear(dim, dim),
#                                nn.ReLU(),
#                                nn.Linear(dim, dim))
#
#    def forward(self):
#
#
#class Transformer(nn.Module):
#    def __init__(self):
#        super().__init__()
#
#    def forward(self):
    
class BVAE_New(pl.LightningModule):
    def __init__(self, energy_fn, energy_loss_fn, reconstruction_weight=0, perceptual_weight=0, energy_weight=0, in_channels = 1, h_dim = 32, lr = 1e-3, batch_size = 100, num_MCMC_iterations = 3, temperature = 0.1, latent_vector_dim = 17):
        super().__init__()
        self.automatic_optimization = False

        self.batch_size = batch_size
        self.latent_vector_dim = latent_vector_dim

        self.lr = lr
        self.vae = VAE(in_channels, h_dim, lr, batch_size)

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

    def MCMC_step(self, batch):
        s = batch

        random_indices = torch.randint(self.latent_vector_dim, size = (self.batch_size, 1), device=self.device)

        s_prime = s.scatter(1, random_indices, 1 - s.gather(1, random_indices))

        s_energy = self.energy_fn(s)
        s_prime_energy = self.energy_fn(s_prime)
        e_matrix = m.e * torch.ones((self.batch_size), device=self.device)
        acceptance_prob_RHS = torch.pow(e_matrix, (s_energy - s_prime_energy) / (self.temperature))        
        acceptance_prob_LHS = torch.ones((self.batch_size), device=self.device)
        acceptance_probability = torch.min(acceptance_prob_LHS, acceptance_prob_RHS)

        s_conjoined = torch.cat((s.unsqueeze(1), s_prime.unsqueeze(1)), dim = 1)

        sample_acceptance = torch.bernoulli(acceptance_probability)

        sample_acceptance = sample_acceptance.unsqueeze(1)

        transitioned_vectors = torch.where(sample_acceptance == 0, s_conjoined[:, 0, :], s_conjoined[:, 1, :])

        return transitioned_vectors

    def training_step(self, batch, batch_idx):
        x, FOM_labels = batch

        opt_VAE= self.optimizers()
        scheduler = self.lr_schedulers()

        #training VAE
        self.toggle_optimizer(opt_VAE)

        logits = self.vae.encode(x)

        probabilities = self.sigmoid(logits)

        #bernoulli sampling
        binary_vector = torch.bernoulli(probabilities)

        original_binary_vector = binary_vector.clone()

        #MCMC
        transitioned_vectors = binary_vector
        for _ in range(self.num_MCMC_iterations):
            transitioned_vectors = self.MCMC_step(transitioned_vectors)

        #straight through gradient copying
        transitioned_vectors_with_gradient = (transitioned_vectors - probabilities).detach() + probabilities
        original_binary_vector_with_gradient = (original_binary_vector - probabilities).detach() + probabilities

        x_hat = self.vae.decode(transitioned_vectors_with_gradient)

        #logging generated images
        sample_imgs_generated = x_hat[:30]
        sample_imgs_original = x[:30]
        gridGenerated = torchvision.utils.make_grid(sample_imgs_generated)
        gridOriginal = torchvision.utils.make_grid(sample_imgs_original)

        if self.global_step % 10 == 0:
            self.logger.experiment.add_image("Generated_images", gridGenerated, self.global_step)
            self.logger.experiment.add_image("Original_images", gridOriginal, self.global_step)

        
        #reconstruction quality
        reconstruction_loss = F.mse_loss(x_hat, x) * self.reconstruction_weight
        perceptual_loss_value = self.perceptual_loss(x_hat, x) * self.perceptual_weight

        #energy correlation
        energy = self.energy_fn(original_binary_vector_with_gradient)
        energy_loss = self.energy_loss_fn(FOM_labels, energy) * self.energy_weight

        total_loss = perceptual_loss_value + reconstruction_loss + energy_loss
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("perceptual_loss", perceptual_loss_value)
        self.log("energy_loss", energy_loss)
        self.log("train_loss", total_loss)

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
        #correlation_loss = -torch.log(-0.5 * (x - 1))

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

class BVAE(pl.LightningModule):
    '''
    Houses both VAE and Discriminator.
    VAE trained with both perceptual and adversarial loss.
    '''
    def __init__(self, a, b, c, d, e, in_channels = 1, h_dim = 32, lr = 1e-3, patch_dim = 32, image_dim = 64, batch_size = 100, num_MCMC_iterations = 3, temperature = 0.1, latent_vector_dim = 17, QUBO_matrix=None, min_energy_vector=-4.25):
        super().__init__()
        self.automatic_optimization = False

        self.image_dim = image_dim #length or width of original square image
        self.patch_dim = patch_dim #length or width of intended square patch
        self.num_patches = (image_dim - patch_dim + 1) ** 2

        self.batch_size = batch_size
        self.latent_vector_dim = latent_vector_dim

        self.lr = lr
        self.vae = VAE(in_channels, h_dim, lr, batch_size)

        self.num_MCMC_iterations = num_MCMC_iterations

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

        self.temperature = temperature

        self.energy_loss_fn = nn.MSELoss()

        self.QUBO_matrix = QUBO_matrix

        self.perceptual_loss = VGGPerceptualLoss()
        self.perceptual_loss.eval()

        self.sigmoid = nn.Sigmoid()

    def MCMC_step(self, batch):
        s = batch

        random_indices = torch.randint(self.latent_vector_dim, size = (self.batch_size, 1), device=self.device)

        s_prime = s.scatter(1, random_indices, 1 - s.gather(1, random_indices))

        s_energy = self.quboEnergy(s, self.QUBO_matrix)
        s_prime_energy = self.quboEnergy(s_prime, self.QUBO_matrix)
        e_matrix = m.e * torch.ones((self.batch_size), device=self.device)
        acceptance_prob_RHS = torch.pow(e_matrix, (s_energy - s_prime_energy) / (self.temperature))        
        acceptance_prob_LHS = torch.ones((self.batch_size), device=self.device)
        acceptance_probability = torch.min(acceptance_prob_LHS, acceptance_prob_RHS)

        s_conjoined = torch.cat((s.unsqueeze(1), s_prime.unsqueeze(1)), dim = 1)

        sample_acceptance = torch.bernoulli(acceptance_probability)

        sample_acceptance = sample_acceptance.unsqueeze(1)

        transitioned_vectors = torch.where(sample_acceptance == 0, s_conjoined[:, 0, :], s_conjoined[:, 1, :])

        return transitioned_vectors

    def training_step(self, batch, batch_idx):
        x, FOM_labels = batch

        opt_VAE= self.optimizers()
        scheduler = self.lr_schedulers()

        #training VAE
        self.toggle_optimizer(opt_VAE)

        logits = self.vae.encode(x)

        probabilities = self.sigmoid(logits)

        #bernoulli sampling
        binary_vector = torch.bernoulli(probabilities)

        original_binary_vector = binary_vector.clone()

        #MCMC
        transitioned_vectors = binary_vector
        for i in range(self.num_MCMC_iterations):
            transitioned_vectors = self.MCMC_step(transitioned_vectors)

        #straight through gradient copying
        transitioned_vectors_with_gradient = (transitioned_vectors - probabilities).detach() + probabilities
        original_binary_vector_with_gradient = (original_binary_vector - probabilities).detach() + probabilities

        x_hat = self.vae.decode(transitioned_vectors_with_gradient)

        #logging generated images
        sample_imgs_generated = x_hat[:30]
        sample_imgs_original = x[:30]
        gridGenerated = torchvision.utils.make_grid(sample_imgs_generated)
        gridOriginal = torchvision.utils.make_grid(sample_imgs_original)

        if self.global_step % 10 == 0:
            self.logger.experiment.add_image("Generated_images", gridGenerated, self.global_step)
            self.logger.experiment.add_image("Original_images", gridOriginal, self.global_step)

        
        reconstruction_loss = F.mse_loss(x_hat, x) * self.a

        perceptual_loss_value = self.perceptual_loss(x_hat, x) * self.b


        #energy correlation
        energy = self.quboEnergy(original_binary_vector_with_gradient, self.QUBO_matrix)
        normalized_FOM = torch.divide((FOM_labels - self.FOM_min), (self.FOM_max - self.FOM_min)) #range [0, 1]
        normalized_FOM = normalized_FOM * self.magnitude_min_energy_vector #to correlate min_energy_vector to best FOM
        energy_loss = self.energy_loss_fn(-normalized_FOM, energy) * self.e


        total_loss = perceptual_loss_value + reconstruction_loss + energy_loss
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("perceptual_loss", perceptual_loss_value)
        self.log("energy_loss", energy_loss)
        self.log("train_loss", total_loss)

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
        self.QUBO_matrix = self.QUBO_matrix.to(self.device)
    def quboEnergy(self, x, H):
        if len(x.shape) == 1 and len(H.shape) == 2:
            return torch.einsum("i,ij,j->", x, H, x)
        elif len(x.shape) == 2 and len(H.shape) == 3:
            return torch.einsum("bi,bij,bj->b", x, H, x)
        elif len(x.shape) == 2 and len(H.shape) == 2:
            return torch.einsum("bi,ij,bj->b", x, H, x)
        else:
            raise ValueError(
                "Invalid shapes for x and H. x must be of shape (batch_size, num_dim) and H must be of shape (batch_size, num_dim, num_dim)."
            )

class VAE(pl.LightningModule):
    '''
    Variational autoencoder with UNet structure.
    '''
    def __init__(self, in_channels = 1, h_dim = 32, lr = 1e-3, batch_size = 100):
        super().__init__()

        self.batch_size = batch_size

        self.lr = lr

        self.attention1E = AttnBlock(h_dim)
        self.attention2E = AttnBlock(h_dim)
        self.resnet1E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet2E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet3E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet4E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet5E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet6E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        #self.resnet7E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
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
                                    nn.Linear(64, 64),
                                    )
        
        '''
        #self.to_mu = nn.Conv2d(h_dim, 1, (1, 1))
        self.to_logits = nn.Sequential(nn.Linear(64, 256),
                                       nn.GroupNorm(8, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 25),
                                       )
        '''
        #self.to_sigma = nn.Conv2d(h_dim, 1, (1, 1))

        self.attention1D = AttnBlock(h_dim)
        self.attention2D = AttnBlock(h_dim)
        #self.resnet0D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet1D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet2D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet3D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet4D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet5D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet6D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        
        '''
        self.to_decoder = nn.Sequential(nn.Linear(25, 256),
                                        nn.GroupNorm(8, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.GroupNorm(8, 64),
                                        nn.ReLU(),
                                        )
        '''
                                        
        self.decoder = nn.Sequential(nn.Linear(64, 64),
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
        return logits
    
    def decode(self, z):
        return self.decoder(z)

class Discriminator(nn.Module):
    def __init__(self, in_channels = 1, h_dim = 64):
        super().__init__()
        self.in_channels = in_channels
        self.h_dim = h_dim
        
        self.layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        

        self.layers = nn.Sequential(
            nn.Conv2d(1, h_dim, kernel_size=4, stride=2, padding=1), #16x16
            nn.GroupNorm(8, h_dim),
            nn.SiLU(),
            nn.Conv2d(h_dim, h_dim * 2, kernel_size=4, stride=2, padding=1), #8x8
            nn.GroupNorm(8, h_dim * 2),
            nn.SiLU(),
            nn.Conv2d(h_dim * 2, h_dim * 4, kernel_size=4, stride=2, padding=1), #4x4
            nn.GroupNorm(8, h_dim * 4),
            nn.SiLU(),
            nn.Conv2d(h_dim * 4, h_dim * 8, kernel_size=4, stride=2, padding=1), #2x2
            nn.GroupNorm(8, h_dim * 8),
            nn.SiLU(),
            nn.Conv2d(h_dim * 8, h_dim * 8, kernel_size=4, stride=2, padding=1), #1x1
            nn.GroupNorm(8, h_dim * 8),
            nn.SiLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(h_dim * 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, self.h_dim * 8)
        x = self.fc(x)
        return x

