import json
from src.models import ef_williams_tf, ef_williams_torch

possible_models = ['ef_williams', 'ef_williams_debug']
possible_loss_fn = ['mae', 'mse']
possible_optimizers = ['adam', 'sgd', 'adagrad', 'adadelta', 'rmsprop']


def get_model(model_str, model_id, param_archi, num_features_text, num_features_image, num_features_audio,
              num_classes, is_tf, max_steps=None, device=None):
    """
    Get model class from string.

    :param model_str: string
    :param model_id: Model id (int)
    :param param_archi: dictionary of 'architecture' hyperparameters of the model chosen (cf. hyperparameters.json)
    :param num_features_text: size of text feature vector (it should be the same at each step)
    :param num_features_image: size of image feature vector (it should be the same at each step)
    :param num_features_audio: size of audio feature vector (it should be the same at each step)
    :param num_classes: Number of classes
    :param is_tf: whether it is TensorFlow or not (boolean)
    :param max_steps: maximum length of sequences (only for TensorFlow)
    :param device: device used, either 'gpu' or 'cpu', only for PyTorch (string)
    :return: Model class
    """

    model_dict = {
        'ef_williams': ef_williams_tf.Model_ef(model_id, param_archi, num_features_text, num_features_image,
                                               num_features_audio, max_steps, num_classes) if is_tf else
                       ef_williams_torch.Model_ef(model_id, param_archi, num_features_text, num_features_image,
                                                  num_features_audio, num_classes).to(device),
        'ef_williams_debug': ef_williams_tf.Model_ef(model_id, param_archi, num_features_text, num_features_image,
                                                     num_features_audio, max_steps, num_classes) if is_tf else
                       ef_williams_torch.Model_ef(model_id, param_archi, num_features_text, num_features_image,
                                                  num_features_audio, num_classes).to(device)  #TODO remove
    }

    if model_str.lower() in model_dict.keys():
        return model_dict[model_str]

    else:
        raise ValueError("Model name '{}' not valid. Please choose among {}.".format(model_str, possible_models))


def get_loss_function(loss_function_str, is_tf):
    """
    Get TensorFlow or PyTorch loss function from string.

    :param loss_function_str: string
    :param is_tf: bool (whether it is TensorFlow or PyTorch)
    :return: TensorFlow or Pytorch loss function
    """
    if is_tf:  # TensorFlow
        loss_fn_dict = {
            'mae': 'mean_absolute_error',
            'mse': 'mean_squared_error'
        }

    else:  # PyTorch
        from torch import nn
        loss_fn_dict = {
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss()
        }

    if loss_function_str.lower() in loss_fn_dict.keys():
        return loss_fn_dict[loss_function_str]
    else:
        raise ValueError("'{}' assigned to key 'loss_function' in 'hyperparameters.json' is invalid."
                         " Please choose among {}.".format(loss_function_str, possible_loss_fn))


def get_optimizer(optimizer_str, learning_rate, is_tf, model=None):
    """
    Get TensorFlow or PyTorch optimizer from string.

    :param optimizer_str: string
    :param learning_rate: float
    :param is_tf: bool (whether it is TensorFlow or PyTorch)
    :param model: a Model object from PyTorch. Let it None if is_tf is True
    :return: TensorFlow or Pytorch optimizer
    """

    if is_tf:  # TensorFlow
        from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
        opt_dict = {
            'adam': Adam(learning_rate=learning_rate),
            'sgd': SGD(learning_rate=learning_rate),
            'adagrad': Adagrad(learning_rate=learning_rate),
            'adadelta': Adadelta(learning_rate=learning_rate),
            'rmsprop': RMSprop(learning_rate=learning_rate)
        }

    else:  # PyTorch
        from torch import optim
        opt_dict = {
            'adam': optim.Adam(model.parameters(), lr=learning_rate),
            'sgd': optim.SGD(model.parameters(), lr=learning_rate),
            'adagrad': optim.Adagrad(model.parameters(), lr=learning_rate),
            'adadelta': optim.Adadelta(model.parameters(), lr=learning_rate),
            'rmsprop': optim.RMSprop(model.parameters(), lr=learning_rate)
        }

    if optimizer_str.lower() in opt_dict.keys():
        return opt_dict[optimizer_str]
    else:
        raise ValueError("'{}' assigned to key 'optimizer' in 'hyperparameters.json' is invalid."
                         " Please choose among {}.".format(optimizer_str, possible_optimizers))


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
    check_value_is_in_list('loss_function', param_training['loss_function'], possible_loss_fn)
    check_value_is_in_list('optimizer', param_training['optimizer'], possible_optimizers)

    return is_tf, param_archi, param_training
