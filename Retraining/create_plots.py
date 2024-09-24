import matplotlib.pyplot as plt
from numpy import linspace
import torch
from torch.utils.data import DataLoader
import pickle

def plot_histogram(experiment_number, degrees):

    for degree in degrees:
        dataset = torch.load(
            f"./Experiment{experiment_number}_Files/{degree}_degree_new_vector_labeled_dataset.pt"
        )
        energy_fn = torch.load(
            f"./Experiment{experiment_number}_Files/{degree}_newly_trained_energy_fn_weights.pt"
        ).cuda()

        dataloader = DataLoader(dataset, batch_size=100, drop_last=True)

        FOMs_list = []
        energies_list = []

        for batch in dataloader:

            with torch.no_grad():

                vectors, FOMs = batch
                vectors = vectors.cuda()
                FOMs = FOMs.cuda()

                energies = energy_fn(vectors)

                energies = torch.squeeze(energies).cpu().numpy()
                FOMs = torch.squeeze(FOMs).cpu().numpy()

                FOMs_list.extend(FOMs)
                energies_list.extend(energies)

        plt.figure()
        plt.hist(FOMs_list)
        plt.xlabel("FOMs")
        plt.ylabel("Frequency")
        plt.title("Histogram of New Dataset Generated After 10 Retraining Iterations")
        plt.savefig(f"./Plots/Histogram_{degree}_degree.png", dpi=500)




def plot_scatter(experiment_number, degrees):

    for degree in degrees:

        dataset = torch.load(
            f"./Experiment{experiment_number}_Files/{degree}_degree_new_vector_labeled_dataset.pt"
        )
        energy_fn = torch.load(
            f"./Experiment{experiment_number}_Files/{degree}_newly_trained_energy_fn_weights.pt"
        ).cuda()

        dataloader = DataLoader(dataset, batch_size=100, drop_last=True)

        FOMs_list = []
        energies_list = []

        for batch in dataloader:

            with torch.no_grad():

                vectors, FOMs = batch
                vectors = vectors.cuda()
                FOMs = FOMs.cuda()

                energies = energy_fn(vectors)

                energies = torch.squeeze(energies).cpu().numpy()
                FOMs = torch.squeeze(FOMs).cpu().numpy()

                FOMs_list.extend(FOMs)
                energies_list.extend(energies)

        # valid_indices = [i for i, value in enumerate(FOMs_list) if 0.0 <= value <= 1.0]

        # filtered_FOMs = [FOMs_list[i] for i in valid_indices]
        # filtered_energies = [energies_list[i] for i in valid_indices]

        plt.figure()
        plt.scatter(FOMs_list, energies_list)
        plt.xlabel("Efficiencies")
        plt.ylabel("Energies")
        plt.savefig(f"./Plots/{degree}_scatter.png", dpi=500)

def plot_training_data(experiment_number, degrees):

    average_FOM_list = []


    for degree in degrees:

        with open(f"./Experiment{experiment_number}_Files/{degree}_degree_training_info.pkl", "rb") as file:
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

    for i, degree in enumerate(degrees):
        if degree == 2:
            experiment_name = "Energy Matching"
        else:
            experiment_name = "Correlational Loss"

        # plt.plot(x_axis, average_FOM_list[i], label=f"Average Efficiencies - Degree {degree}")
        plt.plot(x_axis, average_FOM_list[i], label=f"Average Efficiencies - {experiment_name}")
    # plt.plot(x_axis, average_FOM_list[0], label="Average Efficiencies - Degree 2")
    # plt.plot(x_axis, average_FOM_list[1], label="Average Efficiencies - Degree 3")
    # plt.plot(x_axis, average_FOM_list[2], label="Average Efficiencies - Degree 4")
    plt.legend()
    plt.title(f"Efficiencies over Time")
    plt.savefig(f"./Plots/Average_efficiences_over_time_All_Degrees.png", dpi=500)

def save_training_info(experiment_number, degrees):

    for degree in degrees:

        with open(f"./Experiment{experiment_number}_Files/{degree}_degree_training_info.pkl", "rb") as file:
            experiment_info = pickle.load(file)
            with open(f"./Plots/training_info.txt", "w") as file:
                hyperparameters = experiment_info["Hyperparameters"]
                for key in hyperparameters.keys():
                    file.write(f"{key}: {hyperparameters[key]}\n")



if __name__ == "__main__":
    experiment_number = 4
    degrees = [2, 3]
    # plot_histogram(experiment_number, degrees)
    # plot_scatter(experiment_number, degrees)
    plot_training_data(experiment_number, degrees)
    # save_training_info(experiment_number, degrees)
