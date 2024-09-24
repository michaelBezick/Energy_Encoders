import numpy as np
import matplotlib.pyplot as plt
import pickle

correlational_data = []
for i in range(10):
    with open(f"./Correlational_Loss_{i}_experiment_num_training_info.pkl", "rb") as file:
        correlational_data.append(pickle.load(file))

energy_matching_data = []
for i in range(10):
    with open(f"./Energy_Matching_{i}_experiment_num_training_info.pkl", "rb") as file:
        energy_matching_data.append(pickle.load(file))

# cl_average = np.zeros(10)
# em_average = np.zeros(10)
cl_matrix = np.zeros((10, 10))
em_matrix = np.zeros((10, 10))

for i in range(10):
    cl_run_mean = np.array(correlational_data[i]["Average FOM"])
    # cl_average += cl_run_mean
    #
    em_run_mean = np.array(energy_matching_data[i]["Average FOM"])
    # em_average += em_run_mean

    for j in range(10):
        cl_matrix[i][j] = cl_run_mean[j]
        em_matrix[i][j] = em_run_mean[j]

# print(cl_matrix)
<<<<<<< HEAD
cl_stdevs = np.std(cl_matrix, axis=0)
cl_averages = np.mean(cl_matrix, axis=0)

em_stdevs = np.std(em_matrix, axis=0)
=======
cl_stdevs = np.std(cl_matrix, axis=0) / 3
cl_averages = np.mean(cl_matrix, axis=0)

em_stdevs = np.std(em_matrix, axis=0) / 3
>>>>>>> 28d74e9aadafcd86a9122a59daae71d7bfe76dbc
em_averages = np.mean(em_matrix, axis=0)

x_axis = np.linspace(1, 10, 10)

plt.figure()
plt.errorbar(x_axis, cl_averages, yerr=cl_stdevs, fmt='-o', capsize=5, color='purple', label='Correlational Loss')
plt.errorbar(x_axis, em_averages, yerr=em_stdevs, fmt='-o', capsize=5, color='orange', label='Energy Matching')
<<<<<<< HEAD
# plt.errorbar(x_axis, em_average, yerr=em_stdevs, color='orange', label='Energy Matching')
# plt.plot(x_axis, cl_average, color='purple', label='Correlational Loss')
# plt.plot(x_axis, em_average, color='orange', label='Energy Matching')
=======
>>>>>>> 28d74e9aadafcd86a9122a59daae71d7bfe76dbc
plt.legend()
plt.title("Correlational Loss versus Energy Matching on Average FOM")
plt.xlabel("Retraining Iteration")
plt.ylabel("Average FOM across 10 Separate Runs")
plt.savefig("Retraining.png", dpi=500)

<<<<<<< HEAD
=======

plt.figure()
plt.plot(x_axis, cl_averages, color='purple', label='Correlational Loss')
plt.fill_between(x_axis, cl_averages - cl_stdevs, cl_averages + cl_stdevs, color='purple', alpha=0.2, label='+-1/3 Std Dev')
plt.plot(x_axis, em_averages, color='orange', label='Energy Matching')
plt.fill_between(x_axis, em_averages - em_stdevs, em_averages + em_stdevs, color='orange', alpha=0.2, label='+-1/3 Std Dev')
plt.legend()

plt.title("Correlational Loss versus Energy Matching on Average FOM")
plt.xlabel("Retraining Iteration")
plt.ylabel("Average FOM across 10 Separate Runs")
plt.savefig("Shadow.png", dpi=500)

>>>>>>> 28d74e9aadafcd86a9122a59daae71d7bfe76dbc
exit()
cl_average /= 10
em_average /= 10

x_axis = np.linspace(1, 10, 10)

plt.figure()
plt.plot(x_axis, cl_average, color='purple', label='Correlational Loss')
plt.plot(x_axis, em_average, color='orange', label='Energy Matching')
plt.legend()
plt.title("Correlational Loss versus Energy Matching on Average FOM")
plt.xlabel("Retraining Iteration")
plt.ylabel("Average FOM across 10 Separate Runs")
plt.savefig("Retraining.png", dpi=500)

""""""

# cl_datapoints = []
# em_datapoints = []
# for i in range(10):
#     cl_datapoints.append(np.median(np.array(correlational_data[i]["Average FOM"])))
#     em_datapoints.append(np.median(np.array(energy_matching_data[i]["Average FOM"])))
#
# plt.figure()
#
# x_axis = np.linspace(1, 10, 10)
#
# plt.plot(x_axis, cl_datapoints, color='purple', label='Correlational Loss')
# plt.plot(x_axis, em_datapoints, color='orange', label='Energy Matching')
# plt.legend()
# plt.title("Correlational Loss versus Energy Matching on Median FOM")
# plt.xlabel("Retraining Iteration")
# plt.ylabel("Median FOM across 10 Separate Runs")
# plt.savefig("Median.png", dpi=500)
