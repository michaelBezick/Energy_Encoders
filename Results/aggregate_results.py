import os
import shutil

import matplotlib.pyplot as plt
from functions import (
    Model_Type,
    get_list_of_models,
    get_MCMC_iteratations_from_dir,
    get_model_name_and_type,
)

# get largest FOM
with open("../Evaluate_Model/Experiment_Summary.txt", "r") as file:
    data = file.readline()
    largest_FOM = data.split(" ")[3]


list_of_models = get_list_of_models()  # this is the model path

largest_FOM_Blume_Capel = 0
largest_FOM_Potts = 0
largest_FOM_QUBO = 0

average_FOM_BC = []
average_FOM_BC_index = []
average_FOM_Potts = []
average_FOM_Potts_index = []
average_FOM_QUBO = []
average_FOM_QUBO_index = []

max_FOM_BC = []
max_FOM_BC_index = []
max_FOM_Potts = []
max_FOM_Potts_index = []
max_FOM_QUBO = []
max_FOM_QUBO_index = []


cc_BC = []
cc_BC_index = []
cc_Potts = []
cc_Potts_index = []
cc_QUBO = []
cc_QUBO_index = []

for model_path in list_of_models:
    model_name, model_type = get_model_name_and_type(model_path)
    if not os.path.isdir(model_name):
        os.mkdir(model_name)

    # copying pngs
    shutil.copyfile(
        model_path + "/SampledCorrelation.png", model_name + "/SampledCorrelation.png"
    )
    shutil.copyfile(model_path + "/correlation.png", model_name + "/Correlation.png")

    # getting average and max FOM
    with open(model_path + "/FOM_data.txt", "r") as file:
        average_FOM = float(file.readline().split(" ")[2])
        max_FOM = float(file.readline().split(" ")[2])
        cc = float(file.readline().split(" ")[1])

    if model_type == Model_Type.BLUME_CAPEL:
        average_FOM_BC.append(average_FOM)
        MCMC_iterations = get_MCMC_iteratations_from_dir(model_path)
        average_FOM_BC_index.append(MCMC_iterations)
        max_FOM_BC_index.append(MCMC_iterations)
        max_FOM_BC.append(max_FOM)
        cc_BC.append(cc)
        cc_BC_index.append(MCMC_iterations)

        if max_FOM > largest_FOM_Blume_Capel:
            largest_FOM_Blume_Capel = max_FOM
    elif model_type == Model_Type.POTTS:
        average_FOM_Potts.append(average_FOM)
        MCMC_iterations = get_MCMC_iteratations_from_dir(model_path)
        average_FOM_Potts_index.append(MCMC_iterations)
        max_FOM_Potts_index.append(MCMC_iterations)
        max_FOM_Potts.append(max_FOM)
        cc_Potts.append(cc)
        cc_Potts_index.append(MCMC_iterations)

        if max_FOM > largest_FOM_Potts:
            largest_FOM_Potts = max_FOM
    else:
        average_FOM_QUBO.append(average_FOM)
        MCMC_iterations = get_MCMC_iteratations_from_dir(model_path)
        average_FOM_QUBO_index.append(MCMC_iterations)
        max_FOM_QUBO_index.append(MCMC_iterations)
        max_FOM_QUBO.append(max_FOM)
        cc_QUBO.append(cc)
        cc_QUBO_index.append(MCMC_iterations)

        if max_FOM > largest_FOM_QUBO:
            largest_FOM_QUBO = max_FOM


# creating plot for comparison of average FOM over MCMC iterations
# for all models
tuples_BC = list(zip(average_FOM_BC_index, average_FOM_BC))
tuples_Potts = list(zip(average_FOM_Potts_index, average_FOM_Potts))
tuples_QUBO = list(zip(average_FOM_QUBO_index, average_FOM_QUBO))

sorted_bc = sorted(tuples_BC)
sorted_p = sorted(tuples_Potts)
sorted_q = sorted(tuples_QUBO)

sorted_average_FOM_BC_index = [x[0] for x in sorted_bc]
sorted_average_FOM_BC = [x[1] for x in sorted_bc]

sorted_average_FOM_Potts_index = [x[0] for x in sorted_p]
sorted_average_FOM_Potts = [x[1] for x in sorted_p]

sorted_average_FOM_QUBO_index = [x[0] for x in sorted_q]
sorted_average_FOM_QUBO = [x[1] for x in sorted_q]

plt.figure()
plt.plot(
    sorted_average_FOM_BC_index,
    sorted_average_FOM_BC,
    label="Blume-Capel",
    c="blue",
    marker="o",
)
plt.plot(
    sorted_average_FOM_Potts_index,
    sorted_average_FOM_Potts,
    label="Potts",
    c="red",
    marker="o",
)
plt.plot(
    sorted_average_FOM_QUBO_index,
    sorted_average_FOM_QUBO,
    label="QUBO",
    c="green",
    marker="o",
)

plt.xlabel("MCMC Iterations")
plt.ylabel("Average FOM achieved")
plt.title("Average FOM achieved across models")
plt.legend()

plt.savefig("Average_FOM_Across_Models.png", dpi=300)

tuples_BC = list(zip(max_FOM_BC_index, max_FOM_BC))
tuples_Potts = list(zip(max_FOM_Potts_index, max_FOM_Potts))
tuples_QUBO = list(zip(max_FOM_QUBO_index, max_FOM_QUBO))

sorted_bc = sorted(tuples_BC)
sorted_p = sorted(tuples_Potts)
sorted_q = sorted(tuples_QUBO)

sorted_max_FOM_BC_index = [x[0] for x in sorted_bc]
sorted_max_FOM_BC = [x[1] for x in sorted_bc]

sorted_max_FOM_Potts_index = [x[0] for x in sorted_p]
sorted_max_FOM_Potts = [x[1] for x in sorted_p]

sorted_max_FOM_QUBO_index = [x[0] for x in sorted_q]
sorted_max_FOM_QUBO = [x[1] for x in sorted_q]

plt.figure()
plt.plot(
    sorted_max_FOM_BC_index,
    sorted_max_FOM_BC,
    label="Blume-Capel",
    c="blue",
    marker="o",
)
plt.plot(
    sorted_max_FOM_Potts_index,
    sorted_max_FOM_Potts,
    label="Potts",
    c="red",
    marker="o",
)
plt.plot(
    sorted_max_FOM_QUBO_index,
    sorted_max_FOM_QUBO,
    label="QUBO",
    c="green",
    marker="o",
)

plt.xlabel("MCMC Iterations")
plt.ylabel("Max FOM achieved")
plt.title("Max FOM achieved across models")
plt.legend()

plt.savefig("Max_FOM_Across_Models.png", dpi=300)

tuples_BC = list(zip(cc_BC_index, cc_BC))
tuples_Potts = list(zip(cc_Potts_index, cc_Potts))
tuples_QUBO = list(zip(cc_QUBO_index, cc_QUBO))

sorted_bc = sorted(tuples_BC)
sorted_p = sorted(tuples_Potts)
sorted_q = sorted(tuples_QUBO)

cc_BC_index = [x[0] for x in sorted_bc]
cc_BC = [x[1] for x in sorted_bc]

cc_Potts_index = [x[0] for x in sorted_p]
cc_Potts = [x[1] for x in sorted_p]

cc_QUBO_index = [x[0] for x in sorted_q]
cc_QUBO = [x[1] for x in sorted_q]

plt.figure()
plt.plot(
    cc_BC_index,
    cc_BC,
    label="Blume-Capel",
    c="blue",
    marker="o",
)
plt.plot(
    cc_Potts_index,
    cc_Potts,
    label="Potts",
    c="red",
    marker="o",
)
plt.plot(
    cc_QUBO_index,
    cc_QUBO,
    label="QUBO",
    c="green",
    marker="o",
)

plt.xlabel("MCMC Iterations")
plt.ylabel("Pearson Correlation Coefficient")
plt.title("Pearson Correlation Coefficient Across Models")
plt.legend()

plt.savefig("Pearson_Correlation_Coefficient_Across_Models.png", dpi=300)


with open("Experiment_Summary.txt", "w") as file:
    file.write(f"Max_FOM_Blume-Capel: {largest_FOM_Blume_Capel}\n")
    file.write(f"Max_FOM_Potts: {largest_FOM_Potts}\n")
    file.write(f"Max_FOM_QUBO: {largest_FOM_QUBO}\n")
