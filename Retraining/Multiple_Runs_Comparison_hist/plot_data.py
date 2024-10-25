import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle


plt.rcParams.update({'font.size':14})

a = 10


best_corr_data = []
best_em_data = []


for i in range(a):
    with open(f"./Energy_Matching_{i}_experiment_num_training_info.pkl", "rb") as file:
        best_em_data.append(pickle.load(file))

    with open(f"./Correlational_Loss_{i}_experiment_num_training_info.pkl", "rb") as file:
        best_corr_data.append(pickle.load(file))

corr_FOM_last = []
em_FOM_last = []

for i in range(a):
    corr_n_last = best_corr_data[i]["n last FOM for histogram"]
    # corr_flat = [item for sublist in corr_n_last for item in sublist]

    corr_FOM_last.extend(corr_n_last[-1])


    em_n_last = best_em_data[i]["n last FOM for histogram"]
    # em_flat = [item for sublist in em_n_last for item in sublist]

    em_FOM_last.extend(em_n_last[-1])


plt.figure()
bins = np.linspace(0, 2.0, 100)

plt.hist(em_FOM_last, bins=bins, alpha=0.5, label=r"EM Iteration $\tau_{\text{max}} - 1$", density=True, color='orange')
plt.hist(corr_FOM_last, bins=bins,alpha=0.5, label=r"PearSOL Iteration $\tau_{\text{max}} - 1$", density=True, color='purple')
plt.xlabel("FOM")
plt.ylabel("Frequency")
plt.title("PearSOL vs. EM with PearSOL-regularization\n" + r"Iteration $\tau_{\text{max}} - 1$"+ " (N=10)")
plt.tight_layout()
plt.legend(fontsize="small")
plt.savefig("best_hist.pdf")

cl_loss_matrix = np.zeros((a, 10))
em_loss_matrix = np.zeros((a, 10))

for i in range(a):
    best_corr_mean = np.array(best_corr_data[i]["Average FOM"])
    best_em_mean = np.array(best_em_data[i]["Average FOM"])

    for j in range(10):
        cl_loss_matrix[i][j] = best_corr_mean[j]
        em_loss_matrix[i][j] = best_em_mean[j]

cl_stdevs = np.std(cl_loss_matrix, axis=0) / 3
cl_averages = np.mean(cl_loss_matrix, axis=0)
em_stdevs = np.std(em_loss_matrix, axis=0) / 3
em_averages = np.mean(em_loss_matrix, axis=0)

print(cl_averages)
print(em_averages)

x_axis = np.linspace(1, 10, 10)
plt.figure()
plt.plot(x_axis, cl_averages, color='purple', label="PearSOL")
plt.fill_between(x_axis, cl_averages - cl_stdevs, cl_averages + cl_stdevs, color='purple', alpha=0.2, label=r'$\pm \sigma / 3$')
plt.plot(x_axis, em_averages, color='orange', label="EM")
plt.fill_between(x_axis, em_averages - em_stdevs, em_averages + em_stdevs, color='orange', alpha=0.2, label=r'$\pm \sigma / 3$')
plt.xlabel(r"Retraining Iteration $\tau$")
plt.ylabel("Average FOM")
plt.title("PearSOL vs. EM\nPearSOL Regularization (N=10)")
plt.legend()
plt.tight_layout()
plt.savefig("best_regularization.pdf")



correlational_data_corr_model = []
correlational_data_matching_model = []
for i in range(a):
    with open(f"./Correlational_Loss_Model_Correlational_Loss_Loss_fn_{i}_experiment_num_training_info.pkl", "rb") as file:
        correlational_data_corr_model.append(pickle.load(file))

    with open(f"./Energy_Matching_Model_Correlational_Loss_Loss_fn_{i}_experiment_num_training_info.pkl", "rb") as file:
        correlational_data_matching_model.append(pickle.load(file))

energy_matching_data_corr_model = []
energy_matching_data_matching_model = []
for i in range(a):
    with open(f"./Correlational_Loss_Model_Energy_Matching_Loss_fn_{i}_experiment_num_training_info.pkl", "rb") as file:
        energy_matching_data_corr_model.append(pickle.load(file))
    with open(f"./Energy_Matching_Model_Energy_Matching_Loss_fn_{i}_experiment_num_training_info.pkl", "rb") as file:
        energy_matching_data_matching_model.append(pickle.load(file))


energy_matching_no_affine = []
for i in range(a):
    with open(f"./Energy_Matching_{i}_experiment_num_training_info.pkl", "rb") as file:
        energy_matching_no_affine.append(pickle.load(file))

em_model_no_affine_matrix = np.zeros((a, 10))
em_model_affine_matrix = np.zeros((a, 10))

for i in range(a):
    affine_run_mean = np.array(energy_matching_data_corr_model[i]["Average FOM"])
    no_affine_run_mean = np.array(energy_matching_no_affine[i]["Average FOM"])

    for j in range(10):
        em_model_no_affine_matrix[i][j] = no_affine_run_mean[j]
        em_model_affine_matrix[i][j] = affine_run_mean[j]

# print(cl_matrix)
em_model_affine_stdevs = np.std(em_model_affine_matrix, axis=0) / 3
em_model_affine_averages = np.mean(em_model_affine_matrix, axis=0)
em_model_no_affine_stdevs = np.std(em_model_no_affine_matrix, axis=0) / 3
em_model_no_affine_averages = np.mean(em_model_no_affine_matrix, axis=0)

x_axis = np.linspace(1, 10, 10)
plt.figure()
plt.plot(x_axis, em_model_affine_averages, color='blue', label="EM with Affine")
plt.fill_between(x_axis, em_model_affine_averages - em_model_affine_stdevs, em_model_affine_averages + em_model_affine_stdevs, color='blue', alpha=0.2, label=r'$\pm \sigma / 3$')
plt.plot(x_axis, em_model_no_affine_averages, color='orange', label="EM without Affine")
plt.fill_between(x_axis, em_model_no_affine_averages - em_model_no_affine_stdevs, em_model_no_affine_averages + em_model_no_affine_stdevs, color='orange', alpha=0.2, label=r'$\pm \sigma / 3$')
plt.xlabel(r"Retraining Iteration $\tau$")
plt.ylabel("Average FOM")
plt.title("EM with vs. without Affine Parameters (N=10)")
plt.legend()
plt.tight_layout()
plt.savefig("affine.pdf")


cl_loss_cl_model_matrix = np.zeros((a, 10))
cl_loss_em_model_matrix = np.zeros((a, 10))

for i in range(a):
    cl_loss_cl_model_mean = np.array(correlational_data_corr_model[i]["Average FOM"])
    cl_loss_em_model_mean = np.array(correlational_data_matching_model[i]["Average FOM"])

    for j in range(10):
        cl_loss_cl_model_matrix[i][j] = cl_loss_cl_model_mean[j]
        cl_loss_em_model_matrix[i][j] = cl_loss_em_model_mean[j]

cl_model_stdevs = np.std(cl_loss_cl_model_matrix, axis=0) / 3
cl_model_averages = np.mean(cl_loss_cl_model_matrix, axis=0)
em_model_stdevs = np.std(cl_loss_em_model_matrix, axis=0) / 3
em_model_averages = np.mean(cl_loss_em_model_matrix, axis=0)

plt.figure()
plt.plot(x_axis, cl_model_averages, color='blue', label="PearSOL-Regularization")
plt.fill_between(x_axis, cl_model_averages - cl_model_stdevs, cl_model_averages + cl_model_stdevs, color='blue', alpha=0.2, label=r'$\pm \sigma / 3$')
plt.plot(x_axis, em_model_averages, color='orange', label="EM-Regularization")
plt.fill_between(x_axis, em_model_averages - em_model_stdevs, em_model_averages + em_model_stdevs, color='orange', alpha=0.2, label=r'$\pm \sigma / 3$')
plt.xlabel(r"Retraining Iteration $\tau$")
plt.ylabel("Average FOM")
plt.title("PearSOL with PearSOL vs.\nEM Regularization (N=10)")
plt.legend()
plt.tight_layout()
plt.savefig("regularization.pdf")

# axs[0].plot(x_axis, em_model_cl_averages, color='purple', label='Correlational Loss')
# axs[0].fill_between(x_axis, em_model_cl_averages - em_model_cl_stdevs, em_model_cl_averages + em_model_cl_stdevs, color='purple', alpha=0.2, label=r'$\pm \sigma / 3$')
# axs[0].plot(x_axis, em_model_em_averages, color='orange', label='Energy Matching')
# axs[0].fill_between(x_axis, em_model_em_averages - em_model_em_stdevs, em_model_em_averages + em_model_em_stdevs, color='orange', alpha=0.2, label=r'$\pm \sigma / 3$')
# axs[0].legend(fontsize="small")
# axs[0].set_xlabel("Retraining Iteration")
# axs[0].set_ylabel("Average FOM (N=10)")
# axs[0].set_title("CL vs. EM on Average FOM, EM-bAE\n(N=10)")
#
# axs[1].plot(x_axis, cl_model_cl_averages, color='purple', label='Correlational Loss')
# axs[1].fill_between(x_axis, cl_model_cl_averages - cl_model_cl_stdevs, cl_model_cl_averages + cl_model_cl_stdevs, color='purple', alpha=0.2, label=r'$\pm \sigma / 3$')
# axs[1].plot(x_axis, cl_model_em_averages, color='orange', label='Energy Matching')
# axs[1].fill_between(x_axis, cl_model_em_averages - cl_model_em_stdevs, cl_model_em_averages + cl_model_em_stdevs, color='orange', alpha=0.2, label=r'$\pm \sigma / 3$')
# axs[1].legend(fontsize="small")
# axs[1].set_xlabel("Retraining Iteration")
# axs[1].set_ylabel("Average FOM (N=10)")
# axs[1].set_title("CL vs. EM on Average FOM, CL-bAE\n(N=10)")
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.4)
#
# plt.savefig("all_line_plots.pdf")


og_dataset = torch.load("../../Files/FOM_labels_new.pt")


corr_FOM_total_corr_model = []
corr_FOM_last_corr_model = []
corr_FOM_total_matching_model = []
corr_FOM_last_matching_model = []
for i in range(a):
    corr_n_last = correlational_data_corr_model[i]["n last FOM for histogram"]
    corr_flat = [item for sublist in corr_n_last for item in sublist]

    corr_FOM_total_corr_model.extend(corr_flat)
    corr_FOM_last_corr_model.extend(corr_n_last[-1])

    corr_n_last = correlational_data_matching_model[i]["n last FOM for histogram"]
    corr_flat = [item for sublist in corr_n_last for item in sublist]

    corr_FOM_total_matching_model.extend(corr_flat)
    corr_FOM_last_matching_model.extend(corr_n_last[-1])

plt.figure()
bins = np.linspace(0, 2.0, 100)
plt.hist(og_dataset, bins=bins, alpha=0.5, label="Original Dataset", density=True)
plt.hist(corr_FOM_last_corr_model, bins=bins,alpha=0.5, label=r"Last Iteration $\tau_{\text{max}} - 1$ Vectors", density=True)
plt.xlabel("FOM")
plt.ylabel("Frequency")
plt.title("PearSOL with PearSOL-regularization\n" + r"Last Iteration $\tau_{\text{max}} - 1$"+ " vs. Original Dataset (N=10)")
plt.tight_layout()
plt.legend()
plt.savefig("best vs dataset.pdf")

matching_FOM_total_corr_model = []
matching_FOM_last_corr_model = []
matching_FOM_total_matching_model = []
matching_FOM_last_matching_model = []
for i in range(a):
    matching_n_last = energy_matching_data_corr_model[i]["n last FOM for histogram"]
    matching_flat = [item for sublist in matching_n_last for item in sublist]

    matching_FOM_total_corr_model.extend(matching_flat)
    matching_FOM_last_corr_model.extend(matching_n_last[-1])

    matching_n_last = energy_matching_data_matching_model[i]["n last FOM for histogram"]
    matching_flat = [item for sublist in matching_n_last for item in sublist]

    matching_FOM_total_matching_model.extend(matching_flat)
    matching_FOM_last_matching_model.extend(matching_n_last[-1])


fig, ax = plt.subplots()

# First pair (exp1 vs exp2)
ax.boxplot([matching_FOM_last_matching_model, corr_FOM_last_matching_model], positions=[1, 2], widths=0.6, patch_artist=True, showfliers=False)

# Second pair (exp3 vs exp4), offset by some distance
ax.boxplot([matching_FOM_last_corr_model, corr_FOM_last_corr_model], positions=[4, 5], widths=0.6, patch_artist=True, showfliers=False)

# Customize plot
ax.set_xticks([1.5, 4.5])
# plt.xticks(fontsize=10)
ax.set_xticklabels(['EM vs. PearSOL\n EM-bAE', 'EM vs. PearSOL\nPearSOL-bAE'])
ax.set_ylabel('FOMs')
ax.set_title('EM vs. PearSOL on EM-bAE and PearSOL-bAE\n(N=10)')
plt.savefig("all_boxplot.pdf")

plt.figure()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

bins = np.linspace(0, 2.0, 100)

axs[0].hist(matching_FOM_last_matching_model, bins=bins, alpha=0.5, label='EM', color='orange', density=True)
axs[0].hist(corr_FOM_last_matching_model, bins=bins, alpha=0.5, label='PearSOL', color='purple', density=True)
axs[0].set_title('EM vs. PearSOL with EM-bAE\n(N=10)')
axs[0].set_xlabel('FOM')
axs[0].set_ylabel('Frequency')
axs[0].legend()
axs[0].set_xlim([0, 2.0])

# Second histogram (two sets of data: data2_a and data2_b)
axs[1].hist(matching_FOM_last_corr_model, bins=bins, alpha=0.5, label='EM', color='orange', density=True)
axs[1].hist(corr_FOM_last_corr_model, bins=bins, alpha=0.5, label='PearSOL', color='purple', density=True)
axs[1].set_title('EM vs. PearSOL with PearSOL-bAE\n(N=10)')
axs[1].set_xlabel('FOM')
axs[1].set_ylabel('Frequency')
axs[1].legend()
axs[1].set_xlim([0, 2.0])
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)

plt.savefig("all histograms.pdf")

em_model_cl_matrix = np.zeros((a, 10))
em_model_em_matrix = np.zeros((a, 10))

cl_model_cl_matrix = np.zeros((a, 10))
cl_model_em_matrix = np.zeros((a, 10))

for i in range(a):
    cl_run_mean = np.array(correlational_data_matching_model[i]["Average FOM"])
    em_run_mean = np.array(energy_matching_data_matching_model[i]["Average FOM"])

    for j in range(10):
        em_model_cl_matrix[i][j] = cl_run_mean[j]
        em_model_em_matrix[i][j] = em_run_mean[j]

    cl_run_mean = np.array(correlational_data_corr_model[i]["Average FOM"])
    em_run_mean = np.array(energy_matching_data_corr_model[i]["Average FOM"])

    for j in range(10):
        cl_model_cl_matrix[i][j] = cl_run_mean[j]
        cl_model_em_matrix[i][j] = em_run_mean[j]

# print(cl_matrix)
em_model_cl_stdevs = np.std(em_model_cl_matrix, axis=0) / 3
em_model_cl_averages = np.mean(em_model_cl_matrix, axis=0)
em_model_em_stdevs = np.std(em_model_em_matrix, axis=0) / 3
em_model_em_averages = np.mean(em_model_em_matrix, axis=0)

cl_model_cl_stdevs = np.std(cl_model_cl_matrix, axis=0) / 3
cl_model_cl_averages = np.mean(cl_model_cl_matrix, axis=0)
cl_model_em_stdevs = np.std(cl_model_em_matrix, axis=0) / 3
cl_model_em_averages = np.mean(cl_model_em_matrix, axis=0)

x_axis = np.linspace(1, 10, 10)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(x_axis, em_model_cl_averages, color='purple', label='PearSOL')
axs[0].fill_between(x_axis, em_model_cl_averages - em_model_cl_stdevs, em_model_cl_averages + em_model_cl_stdevs, color='purple', alpha=0.2, label=r'$\pm \sigma / 3$')
axs[0].plot(x_axis, em_model_em_averages, color='orange', label='Energy Matching')
axs[0].fill_between(x_axis, em_model_em_averages - em_model_em_stdevs, em_model_em_averages + em_model_em_stdevs, color='orange', alpha=0.2, label=r'$\pm \sigma / 3$')
axs[0].legend(fontsize="small")
axs[0].set_xlabel(r"Retraining Iteration $\tau$")
axs[0].set_ylabel("Average FOM (N=10)")
axs[0].set_title("PearSOL vs. EM on Average FOM\nEM-bAE (N=10)")

axs[1].plot(x_axis, cl_model_cl_averages, color='purple', label='PearSOL')
axs[1].fill_between(x_axis, cl_model_cl_averages - cl_model_cl_stdevs, cl_model_cl_averages + cl_model_cl_stdevs, color='purple', alpha=0.2, label=r'$\pm \sigma / 3$')
axs[1].plot(x_axis, cl_model_em_averages, color='orange', label='Energy Matching')
axs[1].fill_between(x_axis, cl_model_em_averages - cl_model_em_stdevs, cl_model_em_averages + cl_model_em_stdevs, color='orange', alpha=0.2, label=r'$\pm \sigma / 3$')
axs[1].legend(fontsize="small")
axs[1].set_xlabel(r"Retraining Iteration $\tau$")
axs[1].set_ylabel("Average FOM (N=10)")
axs[1].set_title("PearSOL vs. EM on Average FOM\nPearSOL-bAE (N=10)")
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)

plt.savefig("all_line_plots.pdf")

fig, ax = plt.subplots()

# First pair (exp1 vs exp2)
ax.boxplot([matching_FOM_last_matching_model, corr_FOM_last_matching_model], positions=[1, 2], widths=0.6, patch_artist=True, showfliers=False)

# Second pair (exp3 vs exp4), offset by some distance
ax.boxplot([matching_FOM_last_corr_model, corr_FOM_last_corr_model], positions=[4, 5], widths=0.6, patch_artist=True, showfliers=False)

# Customize plot
ax.set_xticks([1.5, 4.5])
# plt.xticks(fontsize=10)
ax.set_xticklabels(['EM vs. PearSOL\n EM-bAE', 'EM vs. PearSOL\nPearSOL-bAE'])
ax.set_ylabel('FOMs')
ax.set_title('EM vs. PearSOL on EM-bAE and PearSOL-bAE\n(N=10)')
plt.savefig("all_boxplot.pdf")

""""""

plt.figure()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

bins = np.linspace(0, 2.0, 100)

axs[0].hist(matching_FOM_total_matching_model, bins=bins, alpha=0.5, label='EM', color='orange', density=True)
axs[0].hist(corr_FOM_total_matching_model, bins=bins, alpha=0.5, label='PearSOL', color='purple', density=True)
axs[0].set_title('EM vs. PearSOL with EM-bAE\n(N=10)')
axs[0].set_xlabel('FOM')
axs[0].set_ylabel('Frequency')
axs[0].legend()
axs[0].set_xlim([0, 2.0])

# Second histogram (two sets of data: data2_a and data2_b)
axs[1].hist(matching_FOM_total_corr_model, bins=bins, alpha=0.5, label='EM', color='orange', density=True)
axs[1].hist(corr_FOM_total_corr_model, bins=bins, alpha=0.5, label='PearSOL', color='purple', density=True)
axs[1].set_title('EM vs. PearSOL with PearSOL-bAE\n(N=10)')
axs[1].set_xlabel('FOM')
axs[1].set_ylabel('Frequency')
axs[1].legend()
axs[1].set_xlim([0, 2.0])
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)

plt.savefig("all histograms total.pdf")

fig, ax = plt.subplots()

# First pair (exp1 vs exp2)
ax.boxplot([matching_FOM_total_matching_model, corr_FOM_total_matching_model], positions=[1, 2], widths=0.6, patch_artist=True, showfliers=False)

# Second pair (exp3 vs exp4), offset by some distance
ax.boxplot([matching_FOM_total_corr_model, corr_FOM_total_corr_model], positions=[4, 5], widths=0.6, patch_artist=True, showfliers=False)

# Customize plot
ax.set_xticks([1.5, 4.5])
# plt.xticks(fontsize=10)
ax.set_xticklabels(['EM vs. PearSOL\n EM-bAE', 'EM vs. PearSOL\nPearSOL-bAE'])
ax.set_ylabel('FOMs')
ax.set_title('EM vs. PearSOL on EM-bAE and PearSOL-bAE\n(N=10)')
plt.savefig("all_boxplot_total.pdf")
