import json
from src.pytorch_utils.loss_functions import possible_loss_fn_torch
from src.tensorflow_utils.loss_functions import possible_loss_fn_tf


def check_model_name_and_structure(hyperparam_dict, model_name):

    def check_if_key_is_in_list_actual_keys(key, list_actual_keys, root_key):
        if key not in list_actual_keys:
            raise ValueError("'{}' does not have '{}' key. Please create one in 'hyperparameters.json' file."
                              .format(root_key, key, list_actual_keys))

    list_available_models = list(hyperparam_dict['models'].keys())
    if model_name not in list_available_models:
        raise ValueError("Model name '{}' not valid. Please choose among {}.".format(model_name, list_available_models))

    list_keys_from_model = list(hyperparam_dict['models'][model_name].keys())
    check_if_key_is_in_list_actual_keys('is_TensorFlow', list_keys_from_model, model_name)
    check_if_key_is_in_list_actual_keys('architecture', list_keys_from_model, model_name)
    check_if_key_is_in_list_actual_keys('training', list_keys_from_model, model_name)

    list_keys_from_training = list(hyperparam_dict['models'][model_name]['training'].keys())
    check_if_key_is_in_list_actual_keys('num_epochs', list_keys_from_training, "training' in '{}".format(model_name))
    check_if_key_is_in_list_actual_keys('patience', list_keys_from_training, "training' in '{}".format(model_name))
    check_if_key_is_in_list_actual_keys('batch_size', list_keys_from_training, "training' in '{}".format(model_name))
    check_if_key_is_in_list_actual_keys('optimizer', list_keys_from_training, "training' in '{}".format(model_name))
    check_if_key_is_in_list_actual_keys('loss_function', list_keys_from_training, "training' in '{}".format(model_name))
    check_if_key_is_in_list_actual_keys('learning_rate', list_keys_from_training, "training' in '{}".format(model_name))
    check_if_key_is_in_list_actual_keys('val_metric', list_keys_from_training, "training' in '{}".format(model_name))


def check_value_is_number(hp_name, hp_value):

    type_value = type(hp_value)
    if not (type_value is int or type_value is float):
        raise ValueError("The value '{}' assigned to hyperparameter '{}' is not a number. "
                         "Please check 'hyperparameters.json' file.".format(hp_value, hp_name))


def check_value_is_in_list(hp_name, hp_value, list_possible_values):

    if hp_value not in list_possible_values:
        raise ValueError("The value '{}' assigned to hyperparameter '{}' is not among possible values: {}. "
                         "Please check 'hyperparameters.json' file.".format(hp_value, hp_name, list_possible_values))


def get_hyperparam_from_json(model_name):

    with open("hyperparameters.json") as json_file:
        hyperparam = json.load(json_file)

    check_model_name_and_structure(hyperparam, model_name)
    is_tf = (hyperparam['models'][model_name]["is_TensorFlow"])
    check_value_is_in_list("is_TensorFlow", is_tf, [0, 1])
    is_tf = bool(is_tf)
    param_archi = hyperparam['models'][model_name]["architecture"]
    param_training = hyperparam['models'][model_name]["training"]

    # Check hyperparameter values from model training
    check_value_is_number('num_epochs', param_training['num_epochs'])
    check_value_is_number('patience', param_training['patience'])
    check_value_is_number('batch_size', param_training['batch_size'])
    check_value_is_number('learning_rate', param_training['learning_rate'])

    if is_tf:
        check_value_is_in_list('loss_function', param_training['loss_function'], possible_loss_fn_tf)
    else:
        check_value_is_in_list('loss_function', param_training['loss_function'], possible_loss_fn_torch)

    return is_tf, param_archi, param_training
