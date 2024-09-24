import pickle

with open("./Correlational_Loss_3_experiment_num_training_info.pkl", "rb") as file:
    data = pickle.load(file)
    print(f"Average FOM: {data['Average FOM'][-1]}")
    print(f"Average energy: {data['Average energy'][-1]}")
    print(f"Max FOM: {data['Max FOM'][-1]}")
    print(f"Variance of energies: {data['Variance of energies'][-1]}")
    print(f"Variance of FOM: {data['Variance of FOM'][-1]}")
    print(f"Covariance: {data['Covariances'][-1]}")
