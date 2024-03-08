import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np

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

def load_dataset(path):
    dataset = np.expand_dims(np.load(path), 1)
    normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
        
    normalizedDataset = np.multiply(normalizedDataset, 2) - 1

    normalizedDataset = normalizedDataset.astype(np.float32)

    dataset = torch.from_numpy(normalizedDataset)

    labels = torch.from_numpy(np.squeeze(np.load('FOM_labels.npy')))

    labeled_dataset = LabeledDataset(dataset, labels)
    return labeled_dataset
