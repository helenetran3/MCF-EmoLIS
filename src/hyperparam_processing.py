import json


def get_hyperparam_from_json(model_name):

    #TODO check values from hyperparameters.json
    with open("hyperparameters.json") as json_file:
        hyperparam = json.load(json_file)

    list_available_models = list(hyperparam['models'].keys())

    if model_name in list_available_models:
        is_tf = bool(hyperparam['models'][model_name]["is_TensorFlow"])
        param_archi = hyperparam['models'][model_name]["architecture"]
        param_training = hyperparam['models'][model_name]["training"]

    else:
        raise ValueError("Model name '{}' not valid. Please choose among {}.".format(model_name, list_available_models))

    return is_tf, param_archi, param_training
