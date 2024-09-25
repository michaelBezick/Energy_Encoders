import numpy as np
import matplotlib.pyplot as plt
import pickle


def convert_to_hours_per_100(designs_per_minute):
    designs_per_hour = designs_per_minute * 60
    hours_per_design = 1 / designs_per_hour
    hours_per_100_designs = hours_per_design * 100
    return hours_per_100_designs

plt.rcParams.update({'font.size':14})

a = 10

correlational_data = []
for i in range(a):
    with open(f"./Correlational_Loss_{i}_experiment_num_training_info.pkl", "rb") as file:
        correlational_data.append(pickle.load(file))

energy_matching_data = []
for i in range(a):
    with open(f"./Energy_Matching_{i}_experiment_num_training_info.pkl", "rb") as file:
        energy_matching_data.append(pickle.load(file))

lengths = energy_matching_data[0]["Dataset Length"]
increases = [lengths[i] - lengths[i-1] for i in range(1,len(lengths))]
average_increase = sum(increases) / len(increases)
time_per_retraining_iteration = energy_matching_data[0]["Elapsed Time (minutes)"]

designs_per_minute = average_increase / time_per_retraining_iteration
print(designs_per_minute)
print(convert_to_hours_per_100(designs_per_minute))
