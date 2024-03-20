import os
from enum import Enum


class Model_Type(Enum):
    QUBO = 1
    PUBO = 2
    ISING = 3
    BLUME_CAPEL = 4
    POTTS = 5


def get_MCMC_iteratations_from_dir(model_dir):
    print(model_dir)
    name = model_dir.split("/")[4]
    return int(name.split("_")[1])


def get_list_of_models():
    path_list = os.listdir("../Evaluate_Model/Models/")
    models_list = []
    for path in path_list:
        path = "../Evaluate_Model/Models/" + path + "/"
        model_list = os.listdir(path)
        for model_path in model_list:
            if "old_files" in model_path:
                continue
            model_path = (
                path + model_path
            )  # this is the model path without the checkpoint

            models_list.append(model_path)

    return models_list


def get_model_name_and_type(model_dir):
    model_name = model_dir.split("/")[4].split("_")[0]
    "Need to fix this later"
    if model_name == "Blume-Capel":
        model_type = Model_Type.BLUME_CAPEL
    elif model_name == "Potts":
        model_type = Model_Type.POTTS
    else:
        model_type = Model_Type.QUBO

    return model_name, model_type
