import pickle

with open("./Correlational_Loss_9_experiment_num_training_info.pkl", "rb") as file:
    data = pickle.load(file)
    print(data.keys())
    print(data["Covariances"])
