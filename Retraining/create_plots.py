import matplotlib.pyplot as plt
from numpy import linspace
import torch
from torch.utils.data import DataLoader
import pickle


def plot_scatter():
    second_dataset = torch.load(
        "./Experiment1_Files/2_degree_new_vector_labeled_dataset.pt"
    )
    third_dataset = torch.load("./Experiment1_Files/3_degree_new_vector_labeled_dataset.pt")
    fourth_dataset = torch.load(
        "./Experiment1_Files/4_degree_new_vector_labeled_dataset.pt"
    )

    second_energy_fn = torch.load(
        "./Experiment1_Files/2_newly_trained_energy_fn_weights.pt"
    ).cuda()
    third_energy_fn = torch.load(
        "./Experiment1_Files/3_newly_trained_energy_fn_weights.pt"
    ).cuda()
    fourth_energy_fn = torch.load(
        "./Experiment1_Files/4_newly_trained_energy_fn_weights.pt"
    ).cuda()

    second_dataloader = DataLoader(second_dataset, batch_size=100, drop_last=True)
    third_dataloader = DataLoader(third_dataset, batch_size=100, drop_last=True)
    fourth_dataloader = DataLoader(fourth_dataset, batch_size=100, drop_last=True)

    experiment_tuples = [
        (second_energy_fn, second_dataloader),
        (third_energy_fn, third_dataloader),
        (fourth_energy_fn, fourth_dataloader),
    ]


    for i, (energy_fn, loader) in enumerate(experiment_tuples):
        FOMs_list = []
        energies_list = []

        for batch in loader:

            with torch.no_grad():

                vectors, FOMs = batch
                vectors = vectors.cuda()
                FOMs = FOMs.cuda()

                energies = energy_fn(vectors)

                energies = torch.squeeze(energies).cpu().numpy()
                FOMs = torch.squeeze(FOMs).cpu().numpy()

                FOMs_list.extend(FOMs)
                energies_list.extend(energies)

        valid_indices = [i for i, value in enumerate(FOMs_list) if 0.0 <= value <= 1.0]

        filtered_FOMs = [FOMs_list[i] for i in valid_indices]
        filtered_energies = [energies_list[i] for i in valid_indices]

        plt.figure()
        plt.scatter(filtered_FOMs, filtered_energies)
        plt.xlabel("Efficiencies")
        plt.ylabel("Energies")
        plt.savefig(f"./Plots/{i}_scatter.png", dpi=500)

def plot_training_data():

    degrees = [2, 3, 4]

    average_FOM_list = []


    for degree in degrees:

        with open(f"./Experiment1_Files/{degree}_degree_training_info.pkl", "rb") as file:
            experiment_info = pickle.load(file)

            """PLOTTING AVERAGE + MAX FOM"""
            plt.figure()
            plt.xlabel("Retraining Iterations")
            plt.ylabel("Efficiencies")
            x_axis = linspace(1, 10, len(experiment_info["Average FOM"]))
            average_FOM = experiment_info["Average FOM"]
            average_FOM_list.append(average_FOM)
            max_FOM = experiment_info["Max FOM"]
            plt.plot(x_axis, average_FOM, label="Average Efficiencies")
            plt.plot(x_axis, max_FOM, label="Max Efficiency per Retraining Iteration")
            plt.legend()
            plt.title(f"Efficiencies over Time - Degree {degree}")
            plt.savefig(f"./Plots/{degree}_efficiency_over_time.png", dpi=500)


    
    plt.figure()
    plt.xlabel("Retraining Iterations")
    plt.ylabel("Efficiencies")
    plt.plot(x_axis, average_FOM_list[0], label="Average Efficiencies - Degree 2")
    plt.plot(x_axis, average_FOM_list[1], label="Average Efficiencies - Degree 3")
    plt.plot(x_axis, average_FOM_list[2], label="Average Efficiencies - Degree 4")
    plt.legend()
    plt.title(f"Efficiencies over Time")
    plt.savefig(f"./Plots/Average_efficiences_over_time_All_Degrees.png", dpi=500)


if __name__ == "__main__":
    plot_training_data()
