import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polytensor
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from annealing_classes import RNN, Variational_Free_Energy
from Energy_Encoder_Classes import (
    BVAE,
    CorrelationalLoss,
    LabeledDataset,
    LabeledDatasetForVectors,
)
from Functions import (
    add_new_vectors_to_dataset,
    calc_efficiencies_of_new_vectors,
    create_initial_dataset_for_vectors,
    perform_annealing,
    retrain_surrogate_model,
)

number_of_vectors_to_add_per_bin = 1000
num_retraining_iterations = 10
energy_function_retraining_epochs = 800
device = "cuda"
annealing_epochs = 200  # in the graphs 100 seems to be a safe place for convergence
lr = 5e-4
batch_size = 100
annealing_batch_size = 200
warmup_steps = 0
temperature = 1
N_gradient_descent = 5
N_samples = 50
log_step_size = 10
min_energy_repeat_threshold = 100000
vector_length = 64
num_vector_samples = 60
num_images_to_save = 100
annealing_lr = 5e-4

RETRAINING = True
print_vector = False
plot = False
save_vectors = False
RNN_type = "Simple_RNN"
initial_temperature = 1

hyperparameters = {
    "number_of_vectors_to_add_per_bin": number_of_vectors_to_add_per_bin,
    "num_retraining_iterations": num_retraining_iterations,
    "energy_function_retraining_epochs": energy_function_retraining_epochs,
    "device": device,
    "annealing_epochs": annealing_epochs,
    "lr": lr,
    "batch_size": batch_size,
    "annealing_batch_size": annealing_batch_size,
    "warmup_steps": warmup_steps,
    "temperature": temperature,
    "N_gradient_descent": N_gradient_descent,
    "N_samples": N_samples,
    "log_step_size": log_step_size,
    "min_energy_repeat_threshold": min_energy_repeat_threshold,
    "vector_length": vector_length,
    "num_vector_samples": num_vector_samples,
    "num_images_to_save": num_images_to_save,
    "annealing_lr": annealing_lr,
}
print(hyperparameters)

##################################################


def select_top_n(group, n=10):
    if len(group) < n:
        return group
    else:
        return group.nlargest(n, "FOMs")


def load_FOM_model(model_path, weights_path):
    with open(model_path, "r") as file:
        data = file.read()

    FOM_calculator = tf.keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator


FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")


def expand_output(tensor: torch.Tensor):
    x = torch.zeros([tensor.size()[0], 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x


def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))


energy_loss_fn = CorrelationalLoss(1, 1, 1)

num_vars = 64

total_time_list = []
time_per_vector_list = []
list_of_unique_vector_lists = []


num_per_degree = [num_vars]
sample_fn = lambda: torch.randn(1, device="cuda")
terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

terms = polytensor.generators.denseFromSparse(terms, num_vars)
terms.append(torch.randn(num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

second_degree_model = BVAE.load_from_checkpoint(
    "./Models/QUBO_order_2/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)
terms.append(torch.randn(num_vars, num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

third_degree_model = BVAE.load_from_checkpoint(
    "./Models/QUBO_order_3/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)

terms.append(torch.randn(num_vars, num_vars, num_vars, num_vars))
energy_fn = polytensor.DensePolynomial(terms)

fourth_degree_model = BVAE.load_from_checkpoint(
    "./Models/QUBO_order_4/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)

dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

dataset = clamp_output(dataset, 0.5)

labels = torch.load("../Files/FOM_labels_new.pt")

labeled_dataset = LabeledDataset(dataset, labels)
print(f"Original dataset length: {labeled_dataset.__len__()}")

original_dataset_train_loader = DataLoader(
    labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

model_list = [second_degree_model, third_degree_model, fourth_degree_model]

energy_fn_lr_list = [1e-5, 1e-5, 1e-5]
norm_weight_list = [50, 50, 50]
energy_weight_list = [1e-2, 1e-2, 1e-2]

for experiment_number, model in enumerate(model_list):

    model = model.to(device)
    model.scale = model.scale.to(device)
    model.energy_fn.coefficients = model.energy_fn.coefficients.to(device)
    model.sum_of_squares_begin = model.sum_of_squares_begin.to(device)

    initial_vector = torch.bernoulli(torch.ones(batch_size, vector_length) * 0.5).to(
        device
    )

    sigma = initial_vector

    average_energies = []
    temperatures = []

    min_energy = 0
    best_vector = torch.zeros(vector_length).to(device)

    unique_vector_list = []
    unique_vector_set = set()

    delay_temp = 0

    print(f"Experiment number: {experiment_number}")

    new_vector_dataset_labeled = LabeledDatasetForVectors()
    print("INITIALLY CREATING NEW LABELED DATASET FOR VECTORS")
    print("------------------------------")
    sampler = torch.multinomial
    latent_vector_dim = 64
    num_logits = 2
    new_vector_dataset_labeled = create_initial_dataset_for_vectors(
        original_dataset_train_loader, new_vector_dataset_labeled, model
    )

    energy_fn_lr = energy_fn_lr_list[experiment_number]
    surrogate_model_optimizer = torch.optim.Adam(
        params=model.energy_fn.parameters(), lr=energy_fn_lr
    )
    correlational_loss = CorrelationalLoss(10.0, 0.01, 0.0)
    norm_weight = norm_weight_list[experiment_number]
    energy_weight = energy_weight_list[experiment_number]

    surrogate_model_retraining_batch_size = 500

    retraining_information_dict = {}
    retraining_information_dict["Dataset Length"] = []
    retraining_information_dict["Average FOM"] = []
    retraining_information_dict["Max FOM"] = []

    best_images_tuple_list = []

    for i in range(num_images_to_save):
        best_images_tuple_list.append((-100, 1))

    start_time = time.time()

    for retraining_iteration in range(num_retraining_iterations):

        vector_loader_1 = DataLoader(
            new_vector_dataset_labeled,
            batch_size=surrogate_model_retraining_batch_size,
            shuffle=True,
            drop_last=True,
        )

        """IDEA: RETRAIN ENERGY FUNCTION FIRST, THEN PERFORM ANNEALING."""

        print(f"New dataset length: {len(new_vector_dataset_labeled.vectors)}")

        retraining_information_dict["Dataset Length"].append(
            len(new_vector_dataset_labeled.vectors)
        )

        print("------------------------------")
        print(f"PERFORMING RETRAINING, ITERATION: {retraining_iteration}")

        model = retrain_surrogate_model(
            vector_loader_1,
            energy_function_retraining_epochs,
            model,
            correlational_loss,
            energy_fn_lr,
            energy_weight,
            norm_weight,
        )

        """PERFORM ANNEALING"""
        vfa = Variational_Free_Energy(
            model.energy_fn, N_samples=N_samples, batch_size=annealing_batch_size
        )

        vfa = vfa.to(device)

        print("------------------------------")
        print(f"PERFORMING ANNEALING, ITERATION: {retraining_iteration}")

        rnn = RNN(batch_size=annealing_batch_size).cuda()

        unique_vector_list = perform_annealing(
            annealing_batch_size,
            vector_length,
            device,
            annealing_epochs,
            initial_temperature,
            delay_temp,
            vfa,
            rnn,
            num_vector_samples,
            model,
            annealing_lr,
            N_gradient_descent,
        )

        """ANNEALING COMPLETE"""

        """NEED TO CALCULATE ACTUAL EFFICIENCY FOR EACH VECTOR IN UNIQUE VECTOR LIST"""

        decoding_batch_size = 100
        new_vectors_FOM_list, new_designs = calc_efficiencies_of_new_vectors(
            unique_vector_list, device, decoding_batch_size, model, FOM_calculator
        )

        print(
                f"AVERAGE FOM OF NEW VECTORS:{sum(new_vectors_FOM_list[-100:]) / len(new_vectors_FOM_list[-100:])}, Iteration: {retraining_iteration}"
        )

        print(f"MAX FOM IN THIS ITERATION: {max(new_vectors_FOM_list)}")

        retraining_information_dict["Average FOM"].append(
                sum(new_vectors_FOM_list[-100:])
                / len(new_vectors_FOM_list[-100:])
        )
        retraining_information_dict["Max FOM"].append(max(new_vectors_FOM_list))

        new_vector_dataset_labeled = add_new_vectors_to_dataset(
            unique_vector_list, new_vectors_FOM_list, new_vector_dataset_labeled, device
        )

        """NEED TO KEEP MAX VECTORS"""

        for index, FOM_item in enumerate(new_vectors_FOM_list):
            for i in range(len(best_images_tuple_list)):
                compare = best_images_tuple_list[i][0]
                if FOM_item > compare:
                    best_images_tuple_list[0] = (FOM_item, new_designs[index])
                    best_images_tuple_list = sorted(
                        best_images_tuple_list, key=lambda x: x[0]
                    )
                    break

    best_images = torch.zeros(num_images_to_save, 1, 64, 64)

    end_time = time.time()

    for i in range(num_images_to_save):
        best_images[i, :, :, :] = best_images_tuple_list.pop()[1]

    torch.save(best_images, f"highest_FOM_images_{experiment_number + 2}_degree.pt")

    retraining_information_dict["Elapsed Time (minutes)"] = (end_time - start_time) / 60

    retraining_information_dict["Hyperparameters"] = hyperparameters

    with open(f"{experiment_number + 2}_degree_training_info.pkl", "wb") as file:
        pickle.dump(retraining_information_dict, file)

    torch.save(
        new_vector_dataset_labeled,
        f"{experiment_number + 2}_degree_new_vector_labeled_dataset.pt",
    )

    torch.save(
        model.energy_fn, f"{experiment_number + 2}_newly_trained_energy_fn_weights.pt"
    )
