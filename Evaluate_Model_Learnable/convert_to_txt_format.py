import numpy as np
import torch

second = torch.load("./best_topologies_discrete_measurement_2nd_degree.pt").to(torch.uint8)
second = second.flatten(start_dim=2, end_dim=3).squeeze()
third = torch.load("./best_topologies_discrete_measurement_3rd_degree.pt").to(torch.uint8)
third = third.flatten(start_dim=2, end_dim=3).squeeze()
fourth = torch.load("./best_topologies_discrete_measurement_4th_degree.pt").to(torch.uint8)
fourth = fourth.flatten(start_dim=2, end_dim=3).squeeze()

def convert(dataset, filename, num_images):
    for i in range(num_images):
        image = dataset[i, :]
        with open(f"./{filename}/{i}.txt", "w") as file:
            file.write("64 -0.14 0.14\n")
            file.write("64 -0.14 0.14\n")
            file.write("2 0 0.1\n")
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

convert(second, "Second", 100)
convert(third, "Third", 100)
convert(fourth, "Fourth", 100)
