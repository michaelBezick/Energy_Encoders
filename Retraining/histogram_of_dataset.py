import torch
import numpy as np
import tensorflow as tf
import keras
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = torch.load("../Files/FOM_labels_new.pt")


plt.hist(dataset, bins=100)
plt.savefig("histogram.png", dpi=500)
