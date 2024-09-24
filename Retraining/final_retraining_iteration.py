"""IDEA: Now, after the new dataset of vectors has been trained,
perform 1 final long energy function retraining step and then 1 long annealing step"""

import torch
from torch.utils.data import DataLoader
third_degree_vectors = torch.load("./3_degree_new_vector_labeled_dataset.pt")
fourth_degree_vectors = torch.load("./4_degree_new_vector_labeled_dataset.pt")

"""LOAD MODELS"""
second_degree_energy_fn = torch.load("./2_newly_trained_energy_fn_weights.pt")
third_degree_energy_fn = torch.load("./3_newly_trained_energy_fn_weights.pt")
fourth_degree_energy_fn = torch.load("./4_newly_trained_energy_fn_weights.pt")

surrogate_model_retraining_batch_size = 1000


vector_loader_1 = DataLoader(
    new_vector_dataset_labeled,
    batch_size=surrogate_model_retraining_batch_size,
    shuffle=True,
    drop_last=True,
)
