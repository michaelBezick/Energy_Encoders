from compress_dataset import load_dataset_from_binary
import numpy as np
import torch

def to_rgb(dataset):
    return dataset.repeat(1, 3, 1, 1)

loaded_dataset_2nd = load_dataset_from_binary(
    "./compressed_2nd.bin", [20_000, 1, 32, 32]
)
loaded_dataset_3rd = load_dataset_from_binary(
    "./compressed_3rd.bin", [20_000, 1, 32, 32]
)
loaded_dataset_4th = load_dataset_from_binary(
    "./compressed_4th.bin", [20_000, 1, 32, 32]
)


loaded_dataset_2nd = to_rgb(loaded_dataset_2nd).numpy().astype(np.float16)
loaded_dataset_3rd = to_rgb(loaded_dataset_3rd).numpy().astype(np.float16)
loaded_dataset_4th = to_rgb(loaded_dataset_4th).numpy().astype(np.float16)

np.save("dataset_2nd.npy", loaded_dataset_2nd)
np.save("dataset_3rd.npy", loaded_dataset_3rd)
np.save("dataset_4th.npy", loaded_dataset_4th)
