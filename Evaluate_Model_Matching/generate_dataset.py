import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Energy_Encoder_Classes import CorrelationalLoss, BVAE
import polytensor
from convert_to_KID_format import convert_to_KID_and_save

num_images_to_save = 20_000

def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))


def expand_output(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 64, 64])
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

dataset_2nd = torch.load("../Annealing_Learnable/Models/QUBO_order_2/neural_annealing_vectors.pt")
dataset_3rd = torch.load("../Annealing_Learnable/Models/QUBO_order_3/neural_annealing_vectors.pt")
dataset_4th = torch.load("../Annealing_Learnable/Models/QUBO_order_4/neural_annealing_vectors.pt")

energy_loss_fn = CorrelationalLoss(1, 1, 1)
num_vars = 64

num_per_degree = [num_vars]
sample_fn = lambda: torch.randn(1, device="cuda")
terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

terms = polytensor.generators.denseFromSparse(terms, num_vars)
terms.append(torch.randn(num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

second_degree_model = BVAE.load_from_checkpoint(
    "../Annealing_Learnable/Models/QUBO_order_2/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)
terms.append(torch.randn(num_vars, num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

third_degree_model = BVAE.load_from_checkpoint(
    "../Annealing_Learnable/Models/QUBO_order_3/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)

terms.append(torch.randn(num_vars, num_vars, num_vars, num_vars))
energy_fn = polytensor.DensePolynomial(terms)

fourth_degree_model = BVAE.load_from_checkpoint(
    "../Annealing_Learnable/Models/QUBO_order_4/epoch=9999-step=200000.ckpt",
    energy_fn=energy_fn,
    energy_loss_fn=energy_loss_fn,
    h_dim=128,
)

model_list = [second_degree_model, third_degree_model, fourth_degree_model]


batch_size = 100

train_loader_2nd = DataLoader(
    dataset_2nd,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

train_loader_3rd = DataLoader(
    dataset_3rd,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

train_loader_4th = DataLoader(
    dataset_4th,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)
FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")

iteration_num = 1
for train_loader, model in [(train_loader_2nd, second_degree_model), (train_loader_3rd, third_degree_model), (train_loader_4th, fourth_degree_model)]:

    FOM_measurements = []
    best_images_tuple_list = []

    for i in range(num_images_to_save):
        best_images_tuple_list.append((-100, 1))

    iteration_num += 1
    if iteration_num == 2:
        degree = "2nd"
    elif iteration_num == 3:
        degree = "3rd"
    else:
        degree = "4th"

    model = model.cuda()
    for vectors in tqdm(train_loader):

        vectors = vectors.cuda()
        images = model.vae.decode(vectors)
        images = expand_output(clamp_output(images, 0.5), images.size()[0]).cpu()

        FOM = FOM_calculator(torch.permute(images.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy())
        FOM_measurements.extend(FOM.numpy().flatten().tolist())

        for index, FOM_item in enumerate(FOM.numpy().flatten().tolist()):
            for i in range(len(best_images_tuple_list)):
                compare = best_images_tuple_list[i][0]
                if FOM_item > compare:
                    best_images_tuple_list[0] = (FOM_item, images[index, :, :, :])
                    best_images_tuple_list = sorted(
                        best_images_tuple_list, key=lambda x: x[0]
                    )
                    break

    best_images = torch.zeros(num_images_to_save, 1, 64, 64)

    for i in range(num_images_to_save):
        best_images[i, :, :, :] = best_images_tuple_list.pop()[1]

    convert_to_KID_and_save(best_images, f"dataset_{degree}_degree.npy")
    # torch.save(best_images, f"dataset_{degree}_degree.pt")
