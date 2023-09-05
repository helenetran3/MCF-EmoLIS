import os

from tensorflow.keras.models import load_model
from src.hyperparam_processing import get_loss_function
from src.tensorflow_utils.dataset_loader import DatasetTf
from src.pickle_functions import save_with_pickle
from src.model_metrics import compute_loss_value
from src.label_processing import compute_pred_labels


def model_prediction(model, test_dataset, num_test_samples, save_pred, model_id, model_folder):
    """
    Return the predictions of the model on test dataset.

    :param model: The model to evaluate
    :param test_dataset: The TensorFlow dataset for test set
    :param num_test_samples: Number of test samples
    :param save_pred: Whether we save predictions with pickle
    :param model_id: Model id (int)
    :param model_folder: The folder where the result will be saved
    :return: pred_raw: array giving the predictions of the model on test dataset.
    """

    print("\n\n================================= Model Prediction ===========================================\n")
    pred_raw = model.predict(test_dataset, verbose=1, steps=num_test_samples)  # (4654, num_classes)

    if save_pred:
        save_with_pickle(pred_raw, "pred_raw_{}".format(model_id), "predictions", model_folder)

    return pred_raw


def evaluate_tf_model(test_list, param_task, param_training, param_log, max_steps, model_name, model_id, all_scores):
    """
    Evaluate the performance of the best model.

    :param test_list: [x_test, y_test, seg_test]
    :param param_task: dictionary of 'task' hyperparameters (cf. param_task.json)
    :param param_training: dictionary of 'training' hyperparameters (cf. param_training.json)
    :param param_log: dictionary of 'logistics' hyperparameters (cf. param_logistics.json)
    :param max_steps: Maximum length for all sequences in the dataset
    :param model_name: Name of the model currently tested
    :param model_id: Model id (int)
    :param all_scores: list of available presence scores

    :returns loss_function_val, pred_raw, pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom
    """

    predict_neutral_class = param_task['predict_neutral_class']
    threshold_emo_pres = [param_task['threshold_emo_present']]
    label_type = param_task['label_type']
    round_decimals = param_log['round_decimals']
    save_pred = param_log['save_predictions']
    batch_size = param_training['batch_size']
    loss_function = get_loss_function(param_training['loss_function'], is_tf=True)

    # Load best model
    model_folder = os.path.join('results', model_name)
    model_path = os.path.join(model_folder, 'models', "model_{}.h5".format(model_id))
    model = load_model(model_path)

    # Extract x, y and seg_ids for test set
    x_test = test_list[0]  # each element of shape (29, 409)
    y_test = test_list[1]  # each element of shape (1, 6 or 7)
    seg_test = test_list[2]
    num_classes = y_test[0].shape[1]
    num_test_samples = len(y_test)

    # Create TensorFlow test dataset for model prediction and evaluation
    test_dataset = DatasetTf(x_test, y_test, seg_test, max_steps, num_classes, batch_size, train_mode=False).build()

    # Model evaluation
    loss_function_val = compute_loss_value(model, test_dataset, loss_function, round_decimals)

    # Predicted labels
    pred_raw = model_prediction(model, test_dataset, num_test_samples, save_pred, model_id, model_folder)
    pred_scores_coa, pred_classes_pres, pred_classes_dom = \
        compute_pred_labels(pred_raw, all_scores, predict_neutral_class, threshold_emo_pres, save_pred, model_id,
                            num_classes, model_folder, label_type)

    return loss_function_val, pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom
