import json

from src.models import ef_from_mfn, ef_williams_tf, ef_williams_torch, late_fusion, mfn
from src.models.mult import model_mult
# from sklearn.metrics import zero_one_loss, hamming_loss

possible_models = ['ef_from_mfn', 'ef_williams', 'ef_williams_debug', 'late_fusion', 'mfn', 'mult']
possible_loss_fn = ['mae', 'mse', 'bce', 'cce']
possible_val_metrics = ['zero_one_loss', 'hamming_loss', 'rank_loss_ml']
possible_optimizers = ['adam', 'sgd', 'adagrad', 'adadelta', 'rmsprop']


def get_model_module(model_str, is_tf):
    """
    Get model class from string.

    :param model_str: string
    :param is_tf: whether it is TensorFlow or not (boolean)
    :return: Model class
    """

    model_dict = {
        'ef_from_mfn': ef_from_mfn,
        'ef_williams': ef_williams_tf if is_tf else ef_williams_torch,
        'ef_williams_debug': ef_williams_tf if is_tf else ef_williams_torch,
        'late_fusion': late_fusion,
        'mfn': mfn,
        'mult': model_mult
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
            'mse': 'mean_squared_error',
            'bce': 'binary_crossentropy',
            'cce': 'categorical_crossentropy'
        }

    else:  # PyTorch
        from torch import nn
        loss_fn_dict = {
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss(),
            'bce': nn.BCELoss(),
            'cce': nn.CrossEntropyLoss()
        }

    if loss_function_str.lower() in loss_fn_dict.keys():
        return loss_fn_dict[loss_function_str]
    else:
        raise ValueError("'{}' assigned to key 'loss_function' in 'param_training_task.json' is invalid."
                         " Please choose among {}.".format(loss_function_str, possible_loss_fn))


def get_val_metrics(val_metrics_str, is_tf):
    """
    Get TensorFlow or PyTorch loss function from string.

    :param val_metrics_str: string
    :param is_tf: bool (whether it is TensorFlow or PyTorch)
    :return: TensorFlow or Pytorch validation metrics
    """

    def zero_one_loss(output, target):
        if is_tf:
            import tensorflow as tf
            num_diff = tf.math.count_nonzero(output - target, axis=-1)
            loss = tf.where(tf.equal(num_diff, 0), 0, 1)
            loss = tf.cast(loss, tf.float32)
        else:
            import torch
            num_diff = torch.count_nonzero(output - target, dim=-1)
            loss = (num_diff != 0).float()  # 0 if output == target, else 1
            loss = torch.mean(loss, dtype=torch.float32)
        return loss

    def hamming_loss(output, target):
        if is_tf:
            import tensorflow as tf
            num_diff = tf.math.count_nonzero(output - target, axis=-1)
            loss = num_diff / target.shape[-1]
            loss = tf.cast(loss, tf.float32)
        else:
            import torch
            num_diff = torch.count_nonzero(output - target, dim=-1)
            loss_batch = num_diff / target.shape[-1]
            loss = torch.mean(loss_batch, dtype=torch.float32)
            # num_diff = torch.count_nonzero(output - target)
            # loss = num_diff / target.shape[-1]
            # loss = loss / target.shape[0]  # Divide by the batch size
        return loss

    def rank_loss_ml(output, target):  ## Works only for present and dominant classes
        if is_tf:
            import tensorflow as tf
            target, output = tf.squeeze(output), tf.squeeze(target)  # TODO vérifier pour PyTorch
            loss = 0.
            index_pos, index_neg = tf.where(target).numpy(), tf.where(tf.math.abs(1-target)).numpy()
            index_pos, index_neg = list(index_pos), list(index_neg)
            # print("target:", tf.get_static_value(target))
            # print("index_pos:", index_pos)
            # print("index_neg:", index_neg)
            for i_pos_row, i_pos_col in index_pos:
                for i_neg_row, i_neg_col in index_neg:
                    output_pos = output[i_pos_row, i_pos_col]
                    output_neg = output[i_neg_row, i_neg_col]
                    if output_pos < output_neg:
                        loss += 1.
                    if output_pos == output_neg:
                        loss += 0.5
            loss = loss / (len(index_pos) * len(index_neg))
            loss = tf.cast(loss, tf.float32)
        else:
            import torch
            loss = 0  # TODO
        return loss

    val_metrics_dict = {
        'zero_one_loss': zero_one_loss,
        'hamming_loss': hamming_loss,
        'rank_loss_ml': rank_loss_ml
    }

    if val_metrics_str.lower() in val_metrics_dict.keys():
        return val_metrics_dict[val_metrics_str]
    else:
        raise ValueError("'{}' assigned to key 'val_metrics' in 'param_training_task.json' is invalid."
                         " Please choose among {}.".format(val_metrics_str, possible_val_metrics))


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
        raise ValueError("'{}' assigned to key 'optimizer' in 'param_training_task.json' is invalid."
                         " Please choose among {}.".format(optimizer_str, possible_optimizers))


def check_model_name_and_structure(hyperparam_dict, model_name, json_name):

    def check_if_key_is_in_list_actual_keys(keys, hyperparam_subdict, root_key, check_model_in_dict=False):
        list_existing_keys = list(hyperparam_subdict.keys())
        for key in keys:
            if key not in list_existing_keys:
                if check_model_in_dict:
                    raise ValueError(
                        "Model name '{}' not found in {}.json file. Please create a dictionary with '{}' key or"
                        " choose among {}.".format(model_name, model_name, json_name, list_existing_keys))
                else:
                    raise ValueError("'{}' does not have '{}' key. Please create one in '{}.json' file."
                                      .format(root_key, key, json_name))

    def check_value_is_number(hp_name, hp_value):
        type_value = type(hp_value)
        if not (type_value is int or type_value is float):
            raise ValueError("The value '{}' assigned to hyperparameter '{}' is not a number. "
                             "Please check '{}.json' file.".format(hp_value, hp_name, json_name))

    def check_value_is_int_superior_to_min_value(hp_name, hp_value, min_value):
        check_value_is_number(hp_name, hp_value)
        type_value = type(hp_value)
        if type_value is not int:
            raise ValueError("The value '{}' assigned to hyperparameter '{}' is not an integer. "
                             "Please check '{}.json' file.".format(hp_value, hp_name, json_name))
        if hp_value < min_value and hp_value !=-1:
            raise ValueError("The value '{}' assigned to hyperparameter '{}' is not >= to {} (or equal to -1). "
                             "Please check '{}.json' file.".format(hp_value, hp_name, min_value, json_name))

    def check_value_is_in_list(hp_name, hp_value, list_possible_values):
        if hp_value not in list_possible_values:
            raise ValueError("The value '{}' assigned to hyperparameter '{}' is not among possible values: {}. "
                             "Please check '{}.json' file.".format(hp_value, hp_name, list_possible_values, json_name))

    def check_value_is_binary(hp_name, hp_value):
        check_value_is_in_list(hp_name, hp_value, [0, 1])

    def check_set_length_param(hp_value):
        if hp_value != 'avg' and hp_value != 'max' and hp_value != "none" and type(hp_value) != int \
                and (type(hp_value) == int and hp_value == 0):
            raise ValueError("Invalid value for parameter set_length. Please choose value between: 'avg', 'max',"
                             " any strictly positive number, 'none'")

    if json_name == 'param_model':
        check_if_key_is_in_list_actual_keys([model_name], hyperparam_dict['models'], 'models', check_model_in_dict=True)
        hyperparam_model = hyperparam_dict['models'][model_name]
        check_if_key_is_in_list_actual_keys(['is_TensorFlow'], hyperparam_model, model_name)
        check_value_is_binary("is_TensorFlow", hyperparam_model['is_TensorFlow'])

    if json_name == 'param_training_task':

        # Training
        hyperparam_training = hyperparam_dict['training']
        required_keys = ['num_epochs', 'patience', 'batch_size', 'optimizer', 'loss_function', 'learning_rate', 'clip',
                         'when']
        check_if_key_is_in_list_actual_keys(required_keys, hyperparam_training, json_name)
        check_value_is_int_superior_to_min_value('num_epochs', hyperparam_training['num_epochs'], min_value=1)
        check_value_is_int_superior_to_min_value('patience', hyperparam_training['patience'], min_value=0)
        check_value_is_int_superior_to_min_value('batch_size', hyperparam_training['batch_size'], min_value=1)
        check_value_is_int_superior_to_min_value('cross_val_folds', hyperparam_training['cross_val_folds'], min_value=2)
        check_value_is_int_superior_to_min_value('random_state', hyperparam_training['random_state'], min_value=0)
        check_value_is_int_superior_to_min_value('when', hyperparam_training['when'], min_value=0)
        check_value_is_number('learning_rate', hyperparam_training['learning_rate'])
        check_value_is_number('clip', hyperparam_training['clip'])
        check_value_is_in_list('optimizer', hyperparam_training['optimizer'], possible_optimizers)
        check_value_is_in_list('loss_function', hyperparam_training['loss_function'], possible_loss_fn)

        # Task
        hyperparam_task = hyperparam_dict['task']
        required_keys = ['model_name', 'label_type', 'set_length', 'threshold_emo_present', 'predict_neutral_class',
                         'align_to_text', 'image_feature', 'with_custom_split']
        check_if_key_is_in_list_actual_keys(required_keys, hyperparam_task, json_name)
        check_value_is_binary('threshold_emo_present', hyperparam_task['threshold_emo_present'])
        check_value_is_binary('predict_neutral_class', hyperparam_task['predict_neutral_class'])
        check_value_is_binary('align_to_text', hyperparam_task['align_to_text'])
        check_value_is_binary('with_custom_split', hyperparam_task['with_custom_split'])
        check_set_length_param(hyperparam_task['set_length'])
        check_value_is_in_list('model_name', hyperparam_task['model_name'], possible_models)
        check_value_is_in_list('label_type', hyperparam_task['label_type'], ['all', 'coarse', 'present', 'dominant'])
        check_value_is_in_list('image_feature', hyperparam_task['image_feature'], ['facet', 'openface'])

    if json_name == 'param_logistics':
        hyperparam_log = hyperparam_dict['logistics']
        required_keys = ['pickle_name_dataset', 'pickle_name_fold', 'round_decimals', 'plot_confusion_matrix',
                         'save_predictions', 'display_fig']
        check_if_key_is_in_list_actual_keys(required_keys, hyperparam_log, json_name)
        check_value_is_int_superior_to_min_value('round_decimals', hyperparam_log['round_decimals'], min_value=0)
        check_value_is_binary('plot_confusion_matrix', hyperparam_log['plot_confusion_matrix'])
        check_value_is_binary('save_predictions', hyperparam_log['save_predictions'])
        check_value_is_binary('display_fig', hyperparam_log['display_fig'])
        check_value_is_int_superior_to_min_value('num_training_prints', hyperparam_log['num_training_prints'],
                                                 min_value=0)


def get_hyperparam_from_json(json_name, model_name=None):

    with open("config/{}.json".format(json_name)) as json_file:
        hyperparam_dict = json.load(json_file)

    # param_definitions.json has a different structure from the others
    if json_name == "param_definitions":
        if model_name not in list(hyperparam_dict['model'].keys()):
            return None, None
        else:
            return hyperparam_dict['model'][model_name], hyperparam_dict['training']

    # for param_model.json and param_training_task.json
    else:
        check_model_name_and_structure(hyperparam_dict, model_name, json_name)
        if json_name == 'param_model':
            hyperparam_out = hyperparam_dict['models'][model_name]
        elif json_name == 'param_training_task':
            hyperparam_out = (hyperparam_dict['training'], hyperparam_dict['task'])
        elif json_name == 'param_logistics':
            hyperparam_out = hyperparam_dict['logistics']
        else:
            hyperparam_out = None

        return hyperparam_out
