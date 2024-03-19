import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAnnealer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        return F.sigmoid(self.layers(x))


class TensorizedLayer(nn.Module):
    def __init__(self, dim_hidden, dim_sigma):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(dim_hidden, dim_sigma, dim_hidden))
        self.b = nn.Parameter(torch.Tensor(dim_hidden))

        nn.init.xavier_uniform_(self.W)
        nn.init.constant_(self.b, 0.1)

    def forward(self, sigma, hidden):
        x = torch.einsum("bn,anm,bm->ba", sigma, self.W, hidden)
        x = x + self.b
        return x


def convert_to_upper_triangular_QUBO(matrix: torch.Tensor, n):
    new_matrix = torch.zeros_like(matrix)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                new_matrix[i, j] = matrix[i, j]
                continue
            new_matrix[i, j] = matrix[i, j] + matrix[j, i]

    return new_matrix


def convert_to_text(matrix: torch.Tensor, n):
    terms = '{\n\t"cost_function": {\n\t\t"type": "pubo",\n\t\t"version": "1.0",\n\t\t"terms": [\n'
    for i in range(n):
        for j in range(i, n):
            if i == n - 1 and j == n - 1:
                terms += f'\t\t\t{{"c": {matrix[i, j]}, "ids": [{i}, {j}]}}\n'
            else:
                terms += f'\t\t\t{{"c": {matrix[i, j]}, "ids": [{i}, {j}]}},\n'

    terms += "\t\t]\n\t}\n}"
    return terms


class Variational_Free_Energy(nn.Module):
    def __init__(self, energy_fn, N_samples, batch_size=100):
        super().__init__()
        self.energy_fn = energy_fn.to("cuda")
        self.batch_size = batch_size
        self.N_samples = N_samples
        self.epsilon = 1e-10

    def forward(self, sigma_hat_prob: torch.Tensor, temperature):
        energy: torch.Tensor = torch.zeros(self.batch_size, device="cuda")
        for _ in range(self.N_samples):
            if torch.any(sigma_hat_prob < 0):
                raise Exception(f"Outside bounds negative {sigma_hat_prob}")
            if torch.any(sigma_hat_prob > 1):
                raise Exception(f"Outside bounds positive {sigma_hat_prob}")
            sampled = torch.bernoulli(sigma_hat_prob)
            sampled_with_gradient = (sampled - sigma_hat_prob.detach()) + sigma_hat_prob
            energy += torch.squeeze(self.energy_fn(sampled_with_gradient))
            complemented_probabilities = torch.where(
                sampled == 0, 1 - sigma_hat_prob, sigma_hat_prob
            )
            energy += temperature * torch.log(
                torch.prod(complemented_probabilities, dim=1) + self.epsilon
            )

        averaged_local_energy = energy / self.N_samples
        return torch.sum(averaged_local_energy, dim=0) / self.batch_size


class RNN_Concat(nn.Module):
    def __init__(
        self,
        batch_size=100,
        hidden_dim=64,
        vector_length=64,
        device="cuda",
        sigma_dim=2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.vector_length = vector_length
        self.full_dim = hidden_dim + 2
        self.hidden_dim = hidden_dim
        self.device = device
        self.activation_function = nn.ELU()

        self.softmax_weights = nn.ParameterList(
            [nn.Linear(hidden_dim, 2) for _ in range(vector_length)]
        )
        self.softmax = nn.Softmax(dim=1)

        self.layer1 = nn.ParameterList(
            [nn.Linear(self.full_dim, hidden_dim) for _ in range(vector_length)]
        )
        self.layer2 = nn.ParameterList(
            [nn.Linear(self.full_dim, hidden_dim) for _ in range(vector_length)]
        )
        self.layer3 = nn.ParameterList(
            [nn.Linear(self.full_dim, hidden_dim) for _ in range(vector_length)]
        )

    def forward(self, sigma_list: torch.FloatTensor):
        sigma_hat_list = torch.zeros(
            (self.batch_size, self.vector_length), device=self.device
        )
        hidden_state_list = torch.zeros(
            (self.batch_size, self.hidden_dim, self.vector_length), device=self.device
        )
        hidden_state = torch.zeros(
            (self.batch_size, self.hidden_dim), device=self.device
        )

        # computing first set of layers
        for i in range(self.vector_length):
            if i == 0:
                # Null input to model
                sigma = torch.zeros((self.batch_size, 2), device=self.device)
            else:
                sigma = F.one_hot(sigma_list[:, i - 1].type(torch.int64), 2).float()

            hidden_state = self.activation_function(
                self.layer1[i](torch.cat([hidden_state, sigma], dim=1))
            )
            hidden_state_list[:, :, i] = hidden_state

        # computing second set of layers
        sigma = torch.zeros((self.batch_size, 2), device=self.device)
        for i in range(self.vector_length):
            hidden_state = self.activation_function(
                self.layer2[i](torch.cat([hidden_state_list[:, :, i], sigma], dim=1))
            )
            hidden_state_list[:, :, i] = hidden_state

        # computing third set of layers
        sigma = torch.zeros((self.batch_size, 2), device=self.device)
        for i in range(self.vector_length):
            hidden_state = self.activation_function(
                self.layer3[i](torch.cat([hidden_state_list[:, :, i], sigma], dim=1))
            )
            sigma_hat = self.softmax(self.softmax_weights[i](hidden_state))[
                :, 1
            ]  # retrieves the probability of 1
            sigma_hat_list[:, i] = sigma_hat

        return sigma_hat_list


class RNN_Tensorized(nn.Module):
    def __init__(
        self,
        batch_size=100,
        hidden_dim=64,
        vector_length=64,
        device="cuda",
        sigma_dim=2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.vector_length = vector_length
        self.full_dim = hidden_dim + 2
        self.hidden_dim = hidden_dim
        self.device = device
        self.activation_function = nn.ELU()

        self.softmax_weights = nn.ParameterList(
            [nn.Linear(hidden_dim, 2) for _ in range(vector_length)]
        )
        self.softmax = nn.Softmax(dim=1)

        self.tensorized_layer1 = nn.ParameterList(
            [
                TensorizedLayer(dim_hidden=hidden_dim, dim_sigma=sigma_dim)
                for _ in range(vector_length)
            ]
        )
        self.tensorized_layer2 = nn.ParameterList(
            [
                TensorizedLayer(dim_hidden=hidden_dim, dim_sigma=sigma_dim)
                for _ in range(vector_length)
            ]
        )
        self.tensorized_layer3 = nn.ParameterList(
            [
                TensorizedLayer(dim_hidden=hidden_dim, dim_sigma=sigma_dim)
                for _ in range(vector_length)
            ]
        )

    def forward(self, sigma_list: torch.FloatTensor):
        sigma_hat_list = torch.zeros(
            (self.batch_size, self.vector_length), device=self.device
        )
        hidden_state_list = torch.zeros(
            (self.batch_size, self.hidden_dim, self.vector_length), device=self.device
        )
        hidden_state = torch.zeros(
            (self.batch_size, self.hidden_dim), device=self.device
        )

        # computing first set of layers
        for i in range(self.vector_length):
            if i == 0:
                # Null input to model
                sigma = torch.zeros((self.batch_size, 2), device=self.device)
            else:
                sigma = F.one_hot(sigma_list[:, i - 1].type(torch.int64), 2).float()

            hidden_state = self.activation_function(
                self.tensorized_layer1[i](sigma, hidden_state)
            )
            hidden_state_list[:, :, i] = hidden_state
            # sigma_hat = self.softmax(self.softmax_weights[i](hidden_state))[:, 1] #retrieves the probability of 1
            # sigma_hat_list[:, i] = sigma_hat

        # computing second set of layers
        sigma = torch.zeros((self.batch_size, 2), device=self.device)
        for i in range(self.vector_length):
            hidden_state = self.activation_function(
                self.tensorized_layer2[i](sigma, hidden_state_list[:, :, i])
            )
            hidden_state_list[:, :, i] = hidden_state

        # computing third set of layers
        sigma = torch.zeros((self.batch_size, 2), device=self.device)
        for i in range(self.vector_length):
            hidden_state = self.activation_function(
                self.tensorized_layer3[i](sigma, hidden_state_list[:, :, i])
            )
            sigma_hat = self.softmax(self.softmax_weights[i](hidden_state))[
                :, 1
            ]  # retrieves the probability of 1
            sigma_hat_list[:, i] = sigma_hat

        return sigma_hat_list


class RNN(nn.Module):
    def __init__(
        self,
        batch_size=100,
        hidden_dim=64,
        vector_length=64,
        device="cuda",
        sigma_dim=2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.vector_length = vector_length
        self.full_dim = hidden_dim + 2
        self.hidden_dim = hidden_dim
        self.device = device
        self.activation_function = nn.ELU()

        self.hidden_weights = nn.ParameterList(
            [nn.Linear(self.full_dim, hidden_dim) for _ in range(vector_length)]
        )
        self.softmax_weights = nn.ParameterList(
            [nn.Linear(hidden_dim, 2) for _ in range(vector_length)]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sigma_list: torch.FloatTensor):
        sigma_hat_list = torch.zeros(
            (self.batch_size, self.vector_length), device=self.device
        )
        hidden_state = torch.zeros(
            (self.batch_size, self.hidden_dim), device=self.device
        )
        for i in range(self.vector_length):
            if i == 0:
                # Null input to model
                sigma = torch.zeros((self.batch_size, 2), device=self.device)
            else:
                sigma = F.one_hot(sigma_list[:, i - 1].type(torch.int64), 2)

            concatenated_input = torch.cat([hidden_state, sigma], dim=1)
            hidden_state = self.activation_function(
                self.hidden_weights[i](concatenated_input)
            )
            sigma_hat = self.softmax(self.softmax_weights[i](hidden_state))[
                :, 1
            ]  # retrieves the probability of 1
            sigma_hat_list[:, i] = sigma_hat

        return sigma_hat_list


def quboEnergy(x, H):
    """
    Computes the energy for the specified Quadratic Unconstrained Binary Optimization (QUBO) system.

    Parameters:
        x (torch.Tensor) : Tensor of shape (batch_size, num_dim) representing the configuration of the system.
        H (torch.Tensor) : Tensor of shape (batch_size, num_dim, num_dim) representing the QUBO matrix.

    Returns:
        torch.Tensor : The energy for each configuration in the batch.
    """
    if len(x.shape) == 1 and len(H.shape) == 2:
        return torch.einsum("i,ij,j->", x, H, x)
    elif len(x.shape) == 2 and len(H.shape) == 3:
        return torch.einsum("bi,bij,bj->b", x, H, x)
    elif len(x.shape) == 2 and len(H.shape) == 2:
        return torch.einsum("bi,ij,bj->b", x, H, x)
    else:
        raise ValueError(
            "Invalid shapes for x and H. x must be of shape (batch_size, num_dim) and H must be of shape (batch_size, num_dim, num_dim)."
        )
