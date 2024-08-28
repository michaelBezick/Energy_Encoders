import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def txt_to_PNG(filename):
    tensor = torch.zeros((64, 64))
    with open(filename, "r") as file:
        content = file.readlines()
        content = content[3:]
        content = content[:4096]
        row = -1
        for col, line in enumerate(content):
            col = col % 64
            value = int(line[0])
            if col == 0:
                row += 1
            tensor[row, col] = value

        new_filename = filename.split(".")[0]
        plt.imsave(new_filename + ".png", tensor, dpi=500, cmap='gray')
               



txt_to_PNG("Second.txt")
txt_to_PNG("Third.txt")
txt_to_PNG("Fourth.txt")
