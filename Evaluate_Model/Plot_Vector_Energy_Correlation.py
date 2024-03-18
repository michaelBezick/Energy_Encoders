import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Modules.Energy_Encoder_Classes import Model_Type, CorrelationalLoss
from Functions import get_folder_path_from_model_path, get_list_of_models, get_model_name_and_type, get_title_from_model_path, \
                        load_dataset, get_energy_fn, load_energy_functions, BVAE, \
                        get_sampling_vars, load_from_checkpoint, scale_vector_copy_gradient
import torch
from tqdm import tqdm


"""
Code requires that each model be stored as its own directory within ./Models as soley a .ckpt file.
For example, ./Models/Blume-Capel/0_MCMC_0,1_temp/good.ckpt
"""

batch_size = 100
latent_vector_dim = 64
device = "cuda"

models_list = get_list_of_models()

Blume_Capel_energy, Potts_energy, QUBO_energy = load_energy_functions(device)

energy_fn_list = [QUBO_energy, Potts_energy, Blume_Capel_energy]

energy_loss_fn = CorrelationalLoss()

for model_dir in tqdm(models_list):

    model_name, model_type = get_model_name_and_type(model_dir)

    print(model_dir)

    energy_fn = get_energy_fn(model_name, energy_fn_list)

    model = BVAE(energy_fn, energy_loss_fn, h_dim = 128, model_type=model_type)
    model = load_from_checkpoint(model, model_dir)

    num_logits, scale = get_sampling_vars(model)

    #need to plot FOM versus energy
    dataset = load_dataset("../Files/top_0.npy")

    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, drop_last = True)

    FOMs = []
    energies = []

    model = model.eval()
    with torch.no_grad():
        for data in train_loader:
            x, FOM = data
            x = x.to(device)
            scale = scale.to(device)
            logits = model.vae.encode(x)
            probabilities = F.softmax(logits, dim = 2)
            probabilities_condensed = probabilities.view(-1, num_logits)
            sampled_vector = torch.multinomial(probabilities_condensed, 1, True).view(batch_size, latent_vector_dim)
            valid_vector = scale_vector_copy_gradient(sampled_vector, probabilities, scale)
            energy = energy_fn(valid_vector)

            FOMs.extend(FOM.view(-1).detach().cpu().numpy())
            energies.extend(energy.view(-1).detach().cpu().numpy())

    plt.figure()
    plt.scatter(energies, FOMs)
    plot_title = get_title_from_model_path(model_dir)
    plt.title(plot_title)
    model_folder_path = get_folder_path_from_model_path(model_dir)
    plt.savefig(model_folder_path + "/correlation.png")
