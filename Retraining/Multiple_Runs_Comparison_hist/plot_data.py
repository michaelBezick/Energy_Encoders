import numpy as np
import matplotlib.pyplot as plt
import pickle

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

total_corr_FOM = []
corr_FOM_last = []
for i in range(a):
    corr_n_last = correlational_data[i]["n last FOM for histogram"]
    corr_flat = [item for sublist in corr_n_last for item in sublist]
    total_corr_FOM.extend(corr_flat)

    corr_FOM_last.extend(corr_n_last[-1])

total_matching_FOM = []
matching_FOM_last = []
for i in range(a):
    matching_n_last = energy_matching_data[i]["n last FOM for histogram"]
    matching_flat = [item for sublist in matching_n_last for item in sublist]
    total_matching_FOM.extend(matching_flat)

    matching_FOM_last.extend(matching_n_last[-1])


print(f"Correlational average: {sum(total_corr_FOM) / len(total_corr_FOM)}")
print(f"Matching average: {sum(total_matching_FOM) / len(total_matching_FOM)}")

print("------------------------------")

print(f"Correlational average last: {sum(corr_FOM_last) / len(corr_FOM_last)}")
print(f"Matching average last: {sum(matching_FOM_last) / len(matching_FOM_last)}")

plt.figure()
plt.hist(total_matching_FOM, bins=1000, alpha=0.5, color="orange", label="Energy Matching", density=True)
plt.hist(total_corr_FOM, bins=1000, alpha=0.5, color="purple", label="Correlational Loss", density=True)
plt.xlim(0, 1.5)

plt.xlabel("FOM")
plt.ylabel("Normalized Frequency")
plt.title("Correlational Loss versus Energy Matching\n(N = 10)")
plt.legend()
plt.savefig("Histogram.pdf")

plt.figure()
plt.hist(matching_FOM_last, bins=1000, alpha=0.5, color="orange", label="Energy Matching", density=True)
plt.hist(corr_FOM_last, bins=1000, alpha=0.5, color="purple", label="Correlational Loss", density=True)
plt.xlim(0, 1.5)

plt.xlabel("FOM")
plt.ylabel("Normalized Frequency")
plt.title("Correlational Loss versus Energy Matching\n(N = 10)")
plt.legend()
plt.savefig("Last_Histogram.pdf")

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
cl_stdevs = np.std(cl_matrix, axis=0) / 3
cl_averages = np.mean(cl_matrix, axis=0)

em_stdevs = np.std(em_matrix, axis=0) / 3
em_averages = np.mean(em_matrix, axis=0)

x_axis = np.linspace(1, 10, 10)

plt.figure()
plt.errorbar(x_axis, cl_averages, yerr=cl_stdevs, fmt='-o', capsize=5, color='purple', label='Correlational Loss')
plt.errorbar(x_axis, em_averages, yerr=em_stdevs, fmt='-o', capsize=5, color='orange', label='Energy Matching')
plt.legend()
plt.title("Correlational Loss versus Energy Matching on Average FOM")
plt.xlabel("Retraining Iteration")
plt.ylabel("Average FOM across 10 Separate Runs")
plt.savefig("Retraining.pdf", dpi=500)


plt.figure()
plt.plot(x_axis, cl_averages, color='purple', label='Correlational Loss')
plt.fill_between(x_axis, cl_averages - cl_stdevs, cl_averages + cl_stdevs, color='purple', alpha=0.2, label=r'$\pm \sigma / 3$')
plt.plot(x_axis, em_averages, color='orange', label='Energy Matching')
plt.fill_between(x_axis, em_averages - em_stdevs, em_averages + em_stdevs, color='orange', alpha=0.2, label=r'$\pm \sigma / 3$')
plt.legend()

plt.title("Average Sampled FOM\n(N=10)")
plt.xlabel("Retraining Iteration")
plt.ylabel(r"FOM $f(x)$")
plt.savefig("Shadow.pdf")

