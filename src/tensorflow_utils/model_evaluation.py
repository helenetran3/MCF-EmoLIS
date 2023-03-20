import os

from tensorflow.keras.models import load_model
from src.tensorflow_utils.dataset_loader import DatasetTf
from src.pickle_functions import save_with_pickle
from src.model_metrics import compute_loss_value, compute_multilabel_confusion_matrix, plot_multilabel_confusion_matrix
from src.model_metrics import get_and_print_all_metrics
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


def evaluate_model(test_list, batch_size, max_steps, loss_function,
                   model_name, model_id, all_scores, predict_neutral_class, threshold_emo_pres, round_decimals,
                   save_pred):
    """
    Evaluate the performance of the best model.

    :param test_list: [x_test, y_test, seg_test]
    :param batch_size: Batch size for training
    :param max_steps: Maximum length for all sequences in the dataset
    :param loss_function: Loss function
    :param label_type: Type of labels used for model training and evaluation.
    :param model_name: Name of the model currently tested
    :param model_id: Model id (int)
    :param all_scores: list of available presence scores
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param round_decimals: Number of decimals to be rounded for metrics
    :param ext_name: extension name containing info on whether we predict neutral class
    :param save_pred: Whether we save predictions with pickle
    :param plot_confusion_matrix: Whether we save confusion matrices with pickle
    :param display_fig: Whether we display figures

    :returns loss_function_val, pred_raw, pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom
    """

    # Load best model
    model_folder = os.path.join('results', model_name)
    model_path = os.path.join(model_folder, '../models', "model_{}.h5".format(model_id))
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
    pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
        compute_pred_labels(pred_raw, all_scores, predict_neutral_class, threshold_emo_pres, save_pred, model_id,
                            num_classes, model_folder)

    return loss_function_val, pred_raw, pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom
