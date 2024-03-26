import os
import numpy as np

from tensorflow.keras.models import load_model
from src.hyperparam_processing import get_loss_function, get_val_metrics
from src.tensorflow_utils.dataset_loader import DatasetTf
from src.pickle_functions import save_with_pickle
from src.label_processing import compute_pred_labels


def compute_loss_value(model, test_dataset, loss_function, val_metrics, round_decimals):
    """
    Return the loss function value of the model on test dataset.

    :param model: The model to evaluate
    :param test_dataset: The TensorFlow dataset for test set
    :param loss_function: Name of the loss function
    :param val_metrics: Validation metric used for monitoring
    :param round_decimals: number of decimals to be rounded for metrics
    :return: The loss function value
    """

    print("\n\n================================= Model Evaluation ===========================================\n")

    loss_function_val = model.evaluate(test_dataset, verbose=1)
    loss_function_val = np.round(loss_function_val, round_decimals)
    print("Loss function ({}): {}".format(loss_function, loss_function_val[0]))
    print("Validation metric ({}): {}".format(val_metrics, loss_function_val[1]))

    return list(loss_function_val)


def model_prediction(model, test_dataset, num_test_samples, save_pred, ext_name, model_folder):
    """
    Return the predictions of the model on test dataset.

    :param model: The model to evaluate
    :param test_dataset: The TensorFlow dataset for test set
    :param num_test_samples: Number of test samples
    :param save_pred: Whether we save predictions with pickle
    :param ext_name: Extension name (str)
    :param model_folder: The folder where the result will be saved
    :return: pred_raw: array giving the predictions of the model on test dataset.
    """

    print("\n\n================================= Model Prediction ===========================================\n")
    pred_raw = model.predict(test_dataset, verbose=1, steps=num_test_samples)  # (4654, num_classes)

    if save_pred:
        save_with_pickle(pred_raw, "pred_raw_{}".format(ext_name), "predictions", model_folder)

    return pred_raw


def run_one_evaluation(test_list, param_task, param_training, param_log, num_steps, model_name, model_id, all_scores,
                       fold_id=None):

    predict_neutral_class = param_task['predict_neutral_class']
    threshold_emo_pres = [param_task['threshold_emo_present']]
    label_type = param_task['label_type']
    round_decimals = param_log['round_decimals']
    save_pred = param_log['save_predictions']
    batch_size = param_training['batch_size']
    loss_function_str = param_training['loss_function']
    val_metrics_str = param_training['val_metrics']
    ext_model_name = "{}_fold_{}".format(model_id, fold_id) if fold_id is not None else model_id

    # Load best model
    model_folder = os.path.join('results', model_name)
    model_path = os.path.join(model_folder, 'models', "model_{}.h5".format(ext_model_name))
    model = load_model(model_path)

    # Extract x, y and seg_ids for test set
    x_test = test_list[0]  # each element of shape (29, 409)
    y_test = test_list[1]  # each element of shape (1, 6 or 7)
    seg_test = test_list[2]
    num_classes = y_test[0].shape[1]
    num_test_samples = len(y_test)

    # Create TensorFlow test dataset for model prediction and evaluation
    test_dataset = DatasetTf(x_test, y_test, seg_test, num_steps, num_classes, batch_size, train_mode=False).build()

    # Model evaluation
    loss_fn_value = compute_loss_value(model, test_dataset, loss_function_str, val_metrics_str, round_decimals)

    # Predicted labels
    pred_raw = model_prediction(model, test_dataset, num_test_samples, save_pred, ext_model_name, model_folder)
    pred_scores_coa, pred_classes_pres, pred_classes_dom = \
        compute_pred_labels(pred_raw, all_scores, predict_neutral_class, threshold_emo_pres, save_pred, model_id,
                            num_classes, model_folder, label_type)

    # ## For debugging
    # import numpy as np
    # print("pred_raw[0]:", pred_raw[0])
    # print("mean(pred_raw):", np.mean(np.array(pred_raw), axis=0))
    # print("max(pred_raw):", np.max(np.array(pred_raw), axis=0))
    # print("sum(pred_raw > 0.5):", np.sum((np.array(pred_raw) > 0.5), axis=0))
    # if param_task['label_type'] == 'coarse':
    #     print("pred_scores_coa[0]:", pred_scores_coa[0])
    # print("pred_classes_pres[0][0]:", pred_classes_pres[0][0])
    # print("pred_classes_dom[0]:", pred_classes_dom[0])

    return loss_fn_value, pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom


def evaluate_tf_model(test_list, param_task, param_training, param_log, num_steps, model_name, model_id, all_scores,
                      with_cross_valid):
    """
    Evaluate the performance of the best model.
    Returns 5 metrics. if cross-validation, each object is a list of num_folds elements.

    :param test_list: [x_test, y_test, seg_test]
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param param_training: dictionary of 'training' hyperparameters (cf. param_training_task.json)
    :param param_log: dictionary of 'logistics' hyperparameters (cf. param_logistics.json)
    :param num_steps: Maximum length for all sequences in the dataset
    :param model_name: Name of the model currently tested
    :param model_id: Model id (int)
    :param all_scores: list of available presence scores
    :param with_cross_valid: boolean (cross_validation or not)

    :returns model_eval_output, pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom
    """

    if with_cross_valid:
        num_folds = param_training['cross_val_folds']
        model_eval_output, pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom = [], [], [], [], []
        for fold_i in range(num_folds):
            loss_vals_i, pred_raw_i, pred_coa_i, pred_pres_i, pred_dom_i = \
                run_one_evaluation(test_list, param_task, param_training, param_log, num_steps, model_name, model_id,
                                   all_scores, fold_i)
            model_eval_output += [loss_vals_i]  # loss_vals_i = [loss_fn_value, val_met_value]
            pred_raw += [pred_raw_i]
            pred_scores_coa += [pred_coa_i]
            pred_classes_pres += [pred_pres_i]
            pred_classes_dom += [pred_dom_i]
    else:
        model_eval_output, pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            run_one_evaluation(test_list, param_task, param_training, param_log, num_steps, model_name, model_id,
                               all_scores)

    return model_eval_output, pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom
