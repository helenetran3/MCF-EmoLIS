import json
import numpy as np

from src.models import ef_from_mfn, ef_rnn_me2e, ef_trans_me2e, ef_williams_tf, ef_williams_torch, \
    late_fusion, lf_rnn_me2e, lf_trans_me2e, mfn
from src.models.mult import model_mult, model_mult_gml_2
from src.models.me2e import mme2e
# from sklearn.metrics import zero_one_loss, hamming_loss

possible_models = ['ef_from_mfn', 'ef_rnn_me2e', 'ef_trans_me2e', 'ef_williams',
                   'ef_williams_debug', 'late_fusion', 'lf_rnn_me2e', 'lf_trans_me2e',
                   'mfn', 'mme2e', 'mult', 'mult_gml_2']
possible_loss_fn = ['mae', 'mse', 'bce', 'cce', 'wei_bce', 'wei_cce', 'wei_bce_multitask', 'wei_bce_mse_multitask',
                    'wei_bce_gml', 'wei_cce_gml', 'kl_div']
possible_val_metrics = possible_loss_fn + ['zero_one_loss', 'hamming_loss', 'rank_loss_ml', 'rank_loss_gml']
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
        'ef_rnn_me2e': ef_rnn_me2e,
        'ef_trans_me2e': ef_trans_me2e,
        'ef_williams': ef_williams_tf if is_tf else ef_williams_torch,
        'ef_williams_debug': ef_williams_tf if is_tf else ef_williams_torch,
        'late_fusion': late_fusion,
        'lf_rnn_me2e': lf_rnn_me2e,
        'lf_trans_me2e': lf_trans_me2e,
        'mfn': mfn,
        'mme2e': mme2e,
        'mult': model_mult,
        'mult_gml_2': model_mult_gml_2
    }

    if model_str.lower() in model_dict.keys():
        return model_dict[model_str]

    else:
        raise ValueError("Model name '{}' not valid. Please choose among {}.".format(model_str, possible_models))


def get_loss_function(loss_function_str, param_training, is_tf):
    """
    Get TensorFlow or PyTorch loss function from string.

    :param loss_function_str: string
    :param param_training: dictionary of 'training' hyperparameters (cf. param_training_task.json)
    :param is_tf: bool (whether it is TensorFlow or PyTorch)
    :return: TensorFlow or Pytorch loss function
    """

    # Initialization
    loss_fn_dict = {}
    for loss_fn in possible_loss_fn:
        loss_fn_dict[loss_fn] = None

    # TensorFlow
    if is_tf:
        from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, \
            BinaryCrossentropy, CategoricalCrossentropy, KLDivergence
        loss_fn_dict = {
            'mae': MeanAbsoluteError(),
            'mse': MeanSquaredError(),
            'bce': BinaryCrossentropy(),
            'cce': CategoricalCrossentropy(),
            'kl_div': KLDivergence()
        }

    # PyTorch
    else:
        import torch
        from torch import nn
        from torch.autograd import Variable
        pos_weight = torch.tensor([0.86721177, 2.86069315, 3.63110931, 8.98950131, 4.65247525, 11.07615019],
                                  dtype=torch.float32)

        pos_weight_coarse = [[7218, 4370, 2609, 5251], [13533, 2826, 1957, 1132], [14517, 1943, 1362, 1626], [17162, 1592, 492, 202], [15408, 1876, 1098, 1066], [17557, 1156, 548, 187]]
        neg_weight_coarse = [[12230, 15078, 16839, 14197], [5915, 16622, 17491, 18316], [4931, 17505, 18086, 17822], [2286, 17856, 18956, 19246], [4040, 17572, 18350, 18382], [1891, 18292, 18900, 19261]]
        weight_coarse = [[1.6943751731781658, 3.4503432494279176, 6.454197010348793, 2.703675490382784], [0.43707973102785785, 5.881811748053786, 8.937659683188555, 16.180212014134277], [0.3396707308672591, 9.009264024704066, 13.279001468428781, 10.960639606396064], [0.1332012585945694, 11.21608040201005, 38.52845528455285, 95.27722772277228], [0.26220145379023885, 9.366737739872068, 16.712204007285976, 17.24390243902439], [0.10770632796035769, 15.823529411764707, 34.48905109489051, 103.0]]

        def wei_bce_gml(output, target):
            if target.size() != output.size() or target.dim() != 3:  # (num_samples, num_classes, num_scores)
                raise ValueError("Output size does not match with target size. Please check if gml is set to 1 in "
                                 "config/param_model.json.")

            loss = []
            for cl_i, weight_i in enumerate(weight_coarse):
                loss_i = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weight_i))(output[:, cl_i, :], target[:, cl_i, :])
                loss_i = loss_i * pos_weight[cl_i]
                loss += [loss_i]
            loss = torch.mean(torch.Tensor(loss), dtype=torch.float32)
            loss = Variable(loss, requires_grad=True)
            return loss

        def wei_cce_gml(output, target):
            assert(target.size() == output.size())
            assert(target.dim() == 3)  # (num_samples, num_classes, num_scores)

            loss = []
            for cl_i, weight_i in enumerate(weight_coarse):
                loss_i = nn.CrossEntropyLoss(weight=torch.Tensor(weight_i))(output[:, cl_i, :], target[:, cl_i, :])
                loss_i = loss_i * pos_weight[cl_i]
                loss += [loss_i]
            loss = torch.mean(torch.Tensor(loss), dtype=torch.float32)
            loss = Variable(loss, requires_grad=True)
            return loss

        def check_multitask(output, target):
            if not isinstance(output, dict) or \
                    (isinstance(output, dict) and not list(output.keys()) == ['coarse_task', 'fine_task']):
                raise ValueError("Model output does not return a dictionary with keys ['coarse_task', 'fine_task']")
            if not isinstance(target, dict) or \
                    (isinstance(target, dict) and not list(target.keys()) == ['coarse_task', 'fine_task']):
                raise ValueError("Targeted task is not a list of 2 tasks. Please check 'label_type' in "
                                 "config/param_training_task.json")

        def wei_bce_multitask(output, target):
            check_multitask(output, target)
            coef_coa = param_training['coef_coa']
            loss_coa = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output['coarse_task'], target['coarse_task'])
            loss_fin = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output['fine_task'], target['fine_task'])
            loss = coef_coa * loss_coa + (1-coef_coa) * loss_fin
            loss = Variable(torch.Tensor(loss).type(torch.float32), requires_grad=True)
            return loss

        def wei_bce_mse_multitask(output, target):
            check_multitask(output, target)
            coef_coa = param_training['coef_coa']
            loss_coa = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output['coarse_task'], target['coarse_task'])
            loss_fin = nn.MSELoss()(output['fine_task'], target['fine_task'])
            loss = coef_coa * loss_coa + (1-coef_coa) * loss_fin
            loss = Variable(torch.Tensor(loss).type(torch.float32), requires_grad=True)
            return loss

        loss_fn_dict = {
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss(),
            'bce': nn.BCELoss(),
            'cce': nn.CrossEntropyLoss(),
            'wei_bce': nn.BCEWithLogitsLoss(pos_weight=pos_weight),
            'wei_cce': nn.CrossEntropyLoss(weight=pos_weight),
            'wei_bce_multitask': wei_bce_multitask,
            'wei_bce_mse_multitask': wei_bce_mse_multitask,
            'wei_bce_gml': wei_bce_gml,
            'wei_cce_gml': wei_cce_gml,
            'kl_div': nn.KLDivLoss(reduction='batchmean')
        }

    if loss_function_str.lower() in loss_fn_dict.keys():
        loss_fn = loss_fn_dict[loss_function_str]
        if loss_fn is not None:
            return loss_fn
        else:
            lib_name = "TensorFlow" if is_tf else "PyTorch"
            raise ValueError("'{}' assigned to key 'loss_function' in 'param_training_task.json' is not implemented yet"
                             "for {} library. Please choose among {}."
                             .format(loss_function_str, lib_name, possible_loss_fn))
    else:
        raise ValueError("'{}' assigned to key 'loss_function' in 'param_training_task.json' is invalid."
                         " Please choose among {}.".format(loss_function_str, possible_loss_fn))


def get_val_metrics(val_metrics_str, param_training, is_tf):
    """
    Get TensorFlow or PyTorch loss function from string.

    :param val_metrics_str: string
    :param param_training: dictionary of 'training' hyperparameters (cf. param_training_task.json)
    :param is_tf: bool (whether it is TensorFlow or PyTorch)
    :return: TensorFlow or Pytorch validation metrics
    """

    def zero_one_loss(output, target):
        if is_tf:
            import tensorflow as tf
            target, output = output, target
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
            target, output = output, target
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
            target, output = tf.squeeze(output), tf.squeeze(target)
            index_pos, index_neg = tf.where(target).numpy(), tf.where(tf.math.abs(1-target)).numpy()
            # print("target:", tf.get_static_value(target))
            # print("index_pos:", index_pos)
            # print("index_neg:", index_neg)
        else:
            import torch
            index_pos, index_neg = torch.nonzero(target).numpy(), torch.nonzero(torch.abs(target-1)).numpy()
            # print("target:", target)
            # print("index_pos:", index_pos)
            # print("index_neg:", index_neg)

        loss = 0.
        index_pos, index_neg = list(index_pos), list(index_neg)
        # The following line checks if we did not miss any values
        assert(len(index_pos) + len(index_neg) == target.shape[0]*target.shape[1])

        if len(index_pos) * len(index_neg) != 0:
            for i_pos_row, i_pos_col in index_pos:
                for i_neg_row, i_neg_col in index_neg:
                    output_pos = output[i_pos_row, i_pos_col]
                    output_neg = output[i_neg_row, i_neg_col]
                    if output_pos < output_neg:
                        loss += 1.
                    if output_pos == output_neg:
                        loss += 0.5
            loss = loss / (len(index_pos) * len(index_neg))

        if is_tf:
            import tensorflow as tf
            loss = tf.cast(loss, tf.float32)
        else:
            import torch
            loss = torch.tensor(loss, dtype=torch.float32)
        return loss

    def rank_loss_gml(output, target):
        scores = [0, 1/3, 2/3, 1]

        if is_tf:
            import tensorflow as tf
            target, output = tf.squeeze(output), tf.squeeze(target)
            index_scores = [tf.where(tf.equal(target, sc)).numpy() for sc in scores]

        else:
            import torch
            index_scores = [torch.nonzero(target == sc).numpy() for sc in scores]

        index_scores = [list(ind) for ind in index_scores]
        # The following line checks if we did not miss any values
        assert(np.sum([len(ind) for ind in index_scores]) == target.shape[0]*target.shape[1])

        loss, total_values = 0., 0
        for i in range(len(scores)-1):
            for j in range(1, len(scores)):
                if i < j:
                    index_inf, index_sup = index_scores[i], index_scores[j]
                    if len(index_inf) * len(index_sup) != 0:
                        total_values += len(index_inf) * len(index_sup)
                        for i_sup_row, i_sup_col in index_sup:
                            for i_inf_row, i_inf_col in index_inf:
                                output_sup = output[i_sup_row, i_sup_col]
                                output_inf = output[i_inf_row, i_inf_col]
                                if output_sup < output_inf:
                                    loss += 1.
                                if output_sup == output_inf:
                                    loss += 0.5
        loss = loss / total_values

        if is_tf:
            import tensorflow as tf
            loss = tf.cast(loss, tf.float32)
        else:
            import torch
            loss = torch.tensor(loss, dtype=torch.float32)

        return loss

    val_metrics_dict = {
        'zero_one_loss': zero_one_loss,
        'hamming_loss': hamming_loss,
        'rank_loss_ml': rank_loss_ml,
        'rank_loss_gml': rank_loss_gml,
        -1: None
    }

    # Add loss functions to val_metrics_dict
    for loss_fn in possible_loss_fn:
        val_metrics_dict[loss_fn] = get_loss_function(loss_fn, param_training, is_tf)

    if val_metrics_str != -1:
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
                        "Model name '{}' not found in config/param_model.json file. Please create a dictionary with "
                        "'{}' key or choose among {}.".format(model_name, model_name, json_name, list_existing_keys))
                else:
                    raise ValueError("'{}' does not have '{}' key. Please create one in '{}.json' file."
                                      .format(root_key, key, json_name))

    def check_value_is_number(hp_name, hp_value, min_value=None, max_value=None):
        type_value = type(hp_value)
        if not (type_value is int or type_value is float):
            raise ValueError("The value '{}' assigned to hyperparameter '{}' is not a number. "
                             "Please check '{}.json' file.".format(hp_value, hp_name, json_name))
        if min_value is not None and hp_value < min_value and hp_value != -1:
            raise ValueError("The value '{}' assigned to hyperparameter '{}' is smaller than {} (or not equal to -1). "
                             "Please check '{}.json' file.".format(hp_value, hp_name, min_value, json_name))
        if max_value is not None and hp_value > max_value and hp_value != -1:
            raise ValueError("The value '{}' assigned to hyperparameter '{}' is greater than {} (or not equal to -1). "
                             "Please check '{}.json' file.".format(hp_value, hp_name, max_value, json_name))

    def check_value_is_int_superior_to_min_value(hp_name, hp_value, min_value):
        check_value_is_number(hp_name, hp_value, min_value=min_value)
        type_value = type(hp_value)
        if type_value is not int:
            raise ValueError("The value '{}' assigned to hyperparameter '{}' is not an integer. "
                             "Please check '{}.json' file.".format(hp_value, hp_name, json_name))

    def check_value_is_in_list(hp_name, hp_value, list_possible_values):
        def check_value_i_is_in_list(hp_val_i):
            if hp_val_i not in list_possible_values:
                raise ValueError("The value '{}' assigned to hyperparameter '{}' is not among possible values: {}. "
                                 "Please check '{}.json' file.".format(hp_value, hp_name, list_possible_values,
                                                                       json_name))
        if type(hp_value) == list:
            if hp_name == 'label_type':
                if len(hp_value) > 2:
                    raise ValueError("The list '{}' assigned to hyperparameter '{}' contain more than 2 elements. "
                                     "Please check '{}.json' file.".format(hp_value, hp_name, json_name))
                task1, task2 = hp_value[0], hp_value[1]
                if task1 == task2 or ('coarse' in task1 and 'coarse' in task2):
                    raise ValueError("The list '{}' assigned to hyperparameter '{}' contain two values that are "
                                     "assigned to the same type of task. "
                                     "Please check '{}.json' file.".format(hp_value, hp_name, json_name))
            for hp_value_i in hp_value:
                check_value_i_is_in_list(hp_value_i)
        else:
            check_value_i_is_in_list(hp_value)

    def check_value_is_binary(hp_name, hp_value):
        check_value_is_in_list(hp_name, hp_value, [0, 1])

    def check_set_length_param(hp_value):
        if hp_value != 'avg' and hp_value != 'max' and hp_value != "none" and type(hp_value) != int \
                and (type(hp_value) == int and hp_value == 0):
            raise ValueError("Invalid value for parameter set_length. Please choose value between: 'avg', 'max',"
                             " any strictly positive number, 'none'")

    def check_hyperparam_in_form_of_lists(hp_value, str_values, possible_values):
        possible_values = possible_values + [-1] if str_values == 'val_metrics' else possible_values
        if type(hp_value) == list:
            for hp_i in hp_value:
                check_value_is_in_list(str_values, hp_i, possible_values)
        else:
            check_value_is_in_list(str_values, hp_value, possible_values)

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
        check_value_is_number('learning_rate', hyperparam_training['learning_rate'], min_value=0)
        check_value_is_number('clip', hyperparam_training['clip'])
        check_value_is_number('coef_coa', hyperparam_training['coef_coa'], min_value=0, max_value=1)
        check_value_is_in_list('optimizer', hyperparam_training['optimizer'], possible_optimizers)
        check_hyperparam_in_form_of_lists(hyperparam_training['loss_function'], 'loss_function', possible_loss_fn)
        check_hyperparam_in_form_of_lists(hyperparam_training['val_metrics'], 'val_metrics', possible_val_metrics)

        # Task
        hyperparam_task = hyperparam_dict['task']
        required_keys = ['model_name', 'label_type', 'set_length', 'thres_score_pres', 'predict_neutral_class',
                         'align_to_text', 'image_feature', 'with_custom_split']
        check_if_key_is_in_list_actual_keys(required_keys, hyperparam_task, json_name)
        check_value_is_number('thres_score_pres', hyperparam_task['thres_score_pres'])
        check_value_is_number('thres_nan', hyperparam_task['thres_nan'])
        check_value_is_binary('predict_neutral_class', hyperparam_task['predict_neutral_class'])
        check_value_is_binary('align_to_text', hyperparam_task['align_to_text'])
        check_value_is_binary('with_custom_split', hyperparam_task['with_custom_split'])
        check_set_length_param(hyperparam_task['set_length'])
        check_value_is_in_list('model_name', hyperparam_task['model_name'], possible_models)
        check_value_is_in_list('label_type', hyperparam_task['label_type'],
                               ['all', 'coarse', 'coarse_prob', 'coarse_oh', 'present', 'dominant'])
        check_value_is_in_list('image_feature', hyperparam_task['image_feature'], ['facet', 'openface'])
        check_value_is_in_list('replace_nan', hyperparam_task['replace_nan'], ['mean', 'linear_1d', 'linear_2d'])

    if json_name == 'param_logistics':
        hyperparam_log = hyperparam_dict['logistics']
        required_keys = ['pickle_name_dataset', 'root_data_folder', 'round_decimals', 'plot_confusion_matrix',
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
