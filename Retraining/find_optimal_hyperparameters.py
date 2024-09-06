import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Energy_Encoder_Classes import CorrelationalLoss
from torch.optim import Adam
import matplotlib.pyplot as plt


epochs = 10000

"""Hyperparameters"""
optimizer_lr = 1e-5
correlation_weight = 10
norm_weight=10
correlational_loss_weight=1e-2 #in retraining.py I accidentally say "energy_weight"
energy_weight = 0.01 
""""""

writer = SummaryWriter("runs/experiment_1")

third_degree_energy_fn = torch.load("./Experiment1_Files/3_newly_trained_energy_fn_weights.pt").cuda()
third_degree_dataset = torch.load(f"./Experiment1_Files/3_degree_new_vector_labeled_dataset.pt")
third_degree_loader = DataLoader(third_degree_dataset, batch_size=100, shuffle=True, drop_last=True)

correlational_loss = CorrelationalLoss(correlation_weight=correlation_weight, energy_weight=energy_weight, slope_weight=0)

optimizer = Adam(third_degree_energy_fn.parameters(), lr=optimizer_lr)

def calc_norm(terms):
    sum_of_squares = torch.zeros(1, device="cuda")
    for term in terms:
        sum_of_squares = sum_of_squares + torch.sum(term**2)

    return torch.sqrt(sum_of_squares)

i = 0
for epoch in range(epochs):

    energy_list = []
    FOM_list = []
    for batch in third_degree_loader:
        i += 1
        vectors, FOMs = batch
        vectors = vectors.cuda()
        FOMs = FOMs.cuda()

        energies = third_degree_energy_fn(vectors)

        cl = correlational_loss(FOMs, energies) * correlational_loss_weight

        norm = calc_norm(third_degree_energy_fn.coefficients)
        norm_loss = F.mse_loss(norm, torch.ones_like(norm)) * norm_weight

        total_loss = cl + norm_loss

        PCC = correlational_loss.correlation
        average_energy = correlational_loss.average_energy

        if i % 100 == 0:

            writer.add_scalar("total_loss", total_loss.item(), i)
            writer.add_scalar("Pearson Correlation Coefficient", PCC.item(), i)
            writer.add_scalar("Average Energy", average_energy.item(), i)
            writer.add_scalar("Norm", norm.item(), i)
            writer.add_scalar("Correlational Loss", cl.item(), i)
            writer.add_scalar("Norm Loss", norm_loss.item(), i)

        energy_list.extend(torch.squeeze(energies).cpu().detach().numpy())
        FOM_list.extend(torch.squeeze(FOMs).cpu().detach().numpy())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


    plt.figure()
    plt.scatter(FOM_list, energy_list)
    ax = plt.gca()
    plt.xlim(-1, 3)
    writer.add_figure("Scatter", plt.gcf(), epoch)

writer.close()
