from compress_dataset import load_dataset_from_binary
import numpy as np
import torch

loaded_dataset_2nd = load_dataset_from_binary(
    "./compressed_2nd.bin", [20_000, 1, 32, 32]
)
loaded_dataset_3rd = load_dataset_from_binary(
    "./compressed_3rd.bin", [20_000, 1, 32, 32]
)
loaded_dataset_4th = load_dataset_from_binary(
    "./compressed_4th.bin", [20_000, 1, 32, 32]
)

loaded_dataset_2nd = loaded_dataset_2nd.numpy()
loaded_dataset_3rd = loaded_dataset_3rd.numpy()
loaded_dataset_4th = loaded_dataset_4th.numpy()

np.save(loaded_dataset_2nd, "dataset_2nd.npy")
np.save(loaded_dataset_3rd, "dataset_3rd.npy")
np.save(loaded_dataset_4th, "dataset_4th.npy")
