import torch
import tensorflow as tf
from torch.utils.data import DataLoader
import keras

# second = torch.load("./Experiment1_Files/highest_FOM_images_2_degree.pt")
# third = torch.load("./Experiment1_Files/highest_FOM_images_3_degree.pt")
# fourth = torch.load("./Experiment1_Files/highest_FOM_images_4_degree.pt")
#
# # correlational = torch.load("./Experiment4_Files/highest_FOM_images_3_degree.pt")
# correlational = torch.load("./highest_FOM_images_3_degree.pt")
matching = torch.load("./highest_FOM_images_Energy_Matching.pt")

def expand_output(tensor: torch.Tensor):
    x = torch.zeros([tensor.size()[0], 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x

def load_FOM_model(model_path, weights_path):
    with open(model_path, "r") as file:
        data = file.read()

    FOM_calculator = keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator

def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

# second_train_loader = DataLoader(
#     second, batch_size=100, shuffle=True, drop_last=True
# )
#
# third_train_loader = DataLoader(
#     third, batch_size=100, shuffle=True, drop_last=True
# )
#
# fourth_train_loader = DataLoader(
#     fourth, batch_size=100, shuffle=True, drop_last=True
# )
#
# correlational_loader = DataLoader(
#     correlational, batch_size=100, shuffle=True, drop_last=True
# )

matching_loader = DataLoader(
    matching, batch_size=100, shuffle=True, drop_last=True
)


FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")

FOMs = torch.zeros(0)

# list_of_loaders = [second_train_loader, third_train_loader, fourth_train_loader]
# list_of_loaders = [correlational_loader]
list_of_loaders = [matching_loader]

FOM_list = []

for loader in list_of_loaders:

    FOMs_individual = []

    for batch in loader:

        output_expanded = clamp_output(batch, 0.5)

        FOMs = FOM_calculator(
            torch.permute(output_expanded.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy()
        )

        FOMs = FOMs.numpy()

        FOMs = torch.from_numpy(FOMs)
        FOMs_individual.extend(torch.squeeze(FOMs).numpy())

    FOM_list.append(FOMs_individual)

for i, FOMs_individual in enumerate(FOM_list):

    degree = i + 2
    degree = "CORRELATIONAL"

    with open(f"./HELLO.txt", "w") as file:
        for j, FOM in enumerate(FOMs_individual):
            file.write(f"Design {j}: {FOM:.2f}\n")

    exit()

    with open(f"./Designs/{degree}_degree_FOM_list.txt", "w") as file:
        for j, FOM in enumerate(FOMs_individual):
            file.write(f"Design {j}: {FOM:.2f}\n")
