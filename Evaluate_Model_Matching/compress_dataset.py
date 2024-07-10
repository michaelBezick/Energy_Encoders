import torch
from bitarray import bitarray

def expand_output(tensor: torch.Tensor):
    x = torch.zeros([100, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x


def convert_to_binary(dataset: torch.Tensor, save_string):
    if dataset.dtype == torch.float32:
        dataset = dataset.to(torch.int8)

    dataset = torch.flatten(dataset)
    binary_dataset = dataset.tolist()
    binary_bitarray = bitarray(binary_dataset)
    with open(save_string, "wb") as file:
        binary_bitarray.tofile(file)
        print("Saved")


def load_dataset_from_binary(filename, size: list):
    dataset = bitarray()
    with open(filename, "rb") as file:
        dataset.fromfile(file)
        dataset = dataset.tolist()
        dataset = torch.tensor(dataset, dtype=torch.int8)
        dataset = torch.unflatten(dataset, 0, size)

    return dataset

if __name__ == "__main__":
    dataset_2nd = torch.load("./dataset_2nd_degree.pt")
    dataset_3rd = torch.load("./dataset_3rd_degree.pt")
    # dataset_4th = torch.load("./dataset_4th_degree.pt")


    dataset_2nd = dataset_2nd[:, :, 0:32, 0:32]
    dataset_3rd = dataset_3rd[:, :, 0:32, 0:32]
    # dataset_4th = dataset_4th[:, :, 0:32, 0:32]

    convert_to_binary(dataset_2nd, "compressed_2nd.bin")
    convert_to_binary(dataset_3rd, "compressed_3rd.bin")
    # convert_to_binary(dataset_4th, "compressed_4th.bin")
    #
    # # test loading
    # loaded_dataset_2nd = load_dataset_from_binary(
    #     "./compressed_2nd.bin", [20_000, 1, 32, 32]
    # )
    # loaded_dataset_3rd = load_dataset_from_binary(
    #     "./compressed_3rd.bin", [20_000, 1, 32, 32]
    # )
