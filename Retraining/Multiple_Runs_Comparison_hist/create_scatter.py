import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.rcParams.update({'font.size':14})

a = 10

correlational_data_corr_model = []
correlational_data_matching_model = []
for i in range(a):
    with open(f"./Correlational_Loss_Model_Correlational_Loss_Loss_fn_{i}_experiment_num_training_info.pkl", "rb") as file:
        correlational_data_corr_model.append(pickle.load(file))

    with open(f"./Energy_Matching_Model_Correlational_Loss_Loss_fn_{i}_experiment_num_training_info.pkl", "rb") as file:
        correlational_data_matching_model.append(pickle.load(file))

dictionary = correlational_data_corr_model[0]
print(dictionary.keys())
