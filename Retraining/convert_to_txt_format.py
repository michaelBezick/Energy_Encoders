import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader

# second = torch.load("./Experiment1_Files/highest_FOM_images_2_degree.pt").to(torch.uint8).flatten(start_dim=2, end_dim=3).squeeze()
# third = torch.load("./Experiment1_Files/highest_FOM_images_3_degree.pt").to(torch.uint8).flatten(start_dim=2, end_dim=3).squeeze()
# fourth = torch.load("./Experiment1_Files/highest_FOM_images_4_degree.pt").to(torch.uint8).flatten(start_dim=2, end_dim=3).squeeze()
#
# correlational = torch.load("./highest_FOM_images_3_degree.pt").to(torch.uint8).flatten(start_dim=2, end_dim=3).squeeze()
matching = torch.load("./highest_FOM_images_Energy_Matching.pt").to(torch.uint8).flatten(start_dim=2, end_dim=3).squeeze()

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

    FOM_calculator = tf.keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator

def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

# original_dataset_train_loader = DataLoader(
#     fourth, batch_size=100, shuffle=True, drop_last=True
# )
#
# FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")
#
# FOMs = torch.zeros(0)
#
# for batch in original_dataset_train_loader:
#
#     output_expanded = clamp_output(batch, 0.5)
#
#     FOMs = FOM_calculator(
#         torch.permute(output_expanded.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy()
#     )
#
# print(len(FOMs))
# print(sum(FOMs) / len(FOMs))



def convert(dataset, filename, num_images):
    for i in range(num_images):
        image = dataset[i, :]
        with open(f"./{filename}/{i}.txt", "w") as file:
            file.write("64 -0.14 0.14\n")
            file.write("64 -0.14 0.14\n")
            file.write("2 0 0.12\n")
            for pixel in image:
                if pixel.item() == 0:
                    file.write("0\n")
                elif pixel.item() == 1:
                    file.write("1\n")
                else:
                    print("SLKDFJA;DLKAJSD")
                    exit()
            for pixel in image:
                if pixel.item() == 0:
                    file.write("0\n")
                elif pixel.item() == 1:
                    file.write("1\n")

convert(matching, "Matching_3rd", 100)
# convert(second, "Designs/Second", 100)
# convert(third, "Designs/Third", 100)
# convert(fourth, "Designs/Fourth", 100)
