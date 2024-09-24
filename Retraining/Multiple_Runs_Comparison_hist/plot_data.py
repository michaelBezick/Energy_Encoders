import numpy as np
import matplotlib.pyplot as plt
import pickle

with open(f"./Correlational_Loss_0_experiment_num_training_info.pkl", "rb") as file:
    correlational_dict = pickle.load(file)

with open(f"./Energy_Matching_0_experiment_num_training_info.pkl", "rb") as file:
    energy_matching_dict = pickle.load(file)

corr_n_last = correlational_dict["n last FOM for histogram"]
corr_flat = [item for sublist in corr_n_last for item in sublist]

matching_n_last = energy_matching_dict["n last FOM for histogram"]
matching_flat = [item for sublist in matching_n_last for item in sublist]

plt.hist(matching_flat, bins=200, alpha=0.5, color="orange", label="Energy Matching FOM")
plt.hist(corr_flat, bins=200, alpha=0.5, color="purple", label="Correlational Loss FOM")
plt.xlim(0, 1.5)

plt.xlabel("FOM")
plt.ylabel("Frequency")
plt.title("Correlational Loss versus Energy Matching on FOM")
plt.legend()
plt.savefig("Histogram.pdf")
