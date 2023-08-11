from src.pickle_functions import pickle_file_exists, load_from_pickle
from src.dataset_utils import get_dataset_from_sdk, get_fold_ids, process_and_split_dataset, get_maximum_sequence_length_dataset
from src.hyperparam_processing import get_hyperparam_from_json, possible_models
from src.csv_functions import save_model_param_in_csv_file
from src.label_processing import get_available_presence_scores_arr, update_labels
from src.model_metrics import compute_and_save_model_metrics
import argparse

parser = argparse.ArgumentParser(description="SOTA Multimodal Emotion Recognition models using CMU-MOSEI database.")
parser.add_argument('-pnd', '--pickle_name_dataset', type=str,
                    help="Name of the pickle object that will contain the CMU-MOSEI mmdataset.")
parser.add_argument('-pnf', '--pickle_name_fold', type=str,
                    help="Name of the pickle object that will contain the training, validation and test folds.")
parser.add_argument('-t', '--align_to_text', action='store_true',
                    help="Data will be aligned to the textual modality.")
parser.add_argument('-al', '--append_label_to_data', action='store_true',
                    help="Append annotations to the dataset.")
parser.add_argument('-f', '--image_feature', type=str, choices=['facet', 'openface'],
                    help="Image features. Values: facet or openface.")
parser.add_argument('-c', '--with_custom_split', action='store_true',
                    help="Perform custom split on training and validation sets (for more details, cf. Williams et al. "
                         "(2018) paper).")
parser.add_argument('-mn', '--model_name', type=str, choices=possible_models,
                    help="Name of the model currently tested. Values: {}.".format(possible_models))
parser.add_argument('-lt', '--label_type', type=str, choices=['all', 'coarse', 'present', 'dominant'],
                    help="Type of labels used for model training and evaluation. Possible values: all (for all presence"
                         "scores provided by the dataset), coarse (presence score for each emotion among "
                         "[0, 1, 2, 3]), present (multi-label classification), dominant (multi-class classification)")
parser.add_argument('-nc', '--predict_neutral_class', action='store_true',
                    help="Predict neutral class. Only useful for label_type 'present' or 'dominant'.")
parser.add_argument('-tp', '--threshold_emo_present', type=float, nargs='+',
                    help="Threshold at which emotions are considered to be present. Values must be between 0 and 3. "
                         "Note that setting thresholds greater than 0 might lead to no positive true and predicted "
                         "classes and skew classification metrics (F1, precision, recall).")
parser.add_argument('-rd', '--round_decimals', type=int,
                    help="Number of decimals to be rounded for metrics.")
parser.add_argument('-sp', '--save_predictions', action='store_true',
                    help="Save predictions with pickle")
parser.add_argument('-pcm', '--plot_confusion_matrix', action='store_true',
                    help="Plot confusion matrix")
parser.add_argument('-df', '--display_fig', action='store_true',
                    help="Whether we display the figures")
args = parser.parse_args()


def main():

    ext_name = "_n" if args.predict_neutral_class else ""

    # Get data for training, validation and test sets from pickle (split provided by the SDK)
    if pickle_file_exists(args.pickle_name_fold + "_train", "raw_folds", "cmu_mosei"):
        train_list = load_from_pickle(args.pickle_name_fold + "_train", "raw_folds", "cmu_mosei")
        valid_list = load_from_pickle(args.pickle_name_fold + "_valid", "raw_folds", "cmu_mosei")
        test_list = load_from_pickle(args.pickle_name_fold + "_test",  "raw_folds", "cmu_mosei")

    else:
        # Load CMU-MOSEI dataset
        dataset = get_dataset_from_sdk(args.pickle_name_dataset, args.align_to_text, args.append_label_to_data)

        # Get data for training, validation and test sets (split provided by the SDK)
        train_ids, valid_ids, test_ids = get_fold_ids(args.with_custom_split)
        train_list, valid_list, test_list = process_and_split_dataset(dataset,
                                                                      train_ids, valid_ids, test_ids,
                                                                      args.image_feature, args.pickle_name_fold)

    # Get hyperparameters for model architecture and model training + if model available in TensorFlow (tf) or PyTorch
    is_tf, param_archi, param_training = get_hyperparam_from_json(args.model_name)

    all_scores = get_available_presence_scores_arr(train_list, valid_list, test_list)
    max_steps = get_maximum_sequence_length_dataset(train_list, valid_list, test_list)

    # Update labels so that it matches with the targeted task given by args.label_type (all, coarse, present, dominant)
    train_list, valid_list, test_list = \
        update_labels(train_list, valid_list, test_list, args.label_type, all_scores,
                      args.predict_neutral_class, args.threshold_emo_present, ext_name)

    # Save model parameters and return model id
    model_id = save_model_param_in_csv_file(param_archi, param_training, args.predict_neutral_class,
                                            args.label_type, args.model_name, is_tf)

    # Model training
    if is_tf:
        from src.tensorflow_utils.model_training import train_tf_model
        from src.tensorflow_utils.model_evaluation import evaluate_tf_model

        train_tf_model(train_list, valid_list, test_list,
                       param_archi, param_training, max_steps, args.model_name, model_id, args.predict_neutral_class,
                       args.display_fig, args.label_type)

        loss_function_val, pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            evaluate_tf_model(test_list, param_training, max_steps, args.model_name, model_id, all_scores,
                              args.predict_neutral_class, args.threshold_emo_present, args.label_type,
                              args.round_decimals, args.save_predictions)

    else:
        import torch
        from src.pytorch_utils.model_training import train_torch_model
        from src.pytorch_utils.model_evaluation import evaluate_torch_model

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\nUsing {device} device")

        train_torch_model(train_list, valid_list, test_list,
                          param_archi, param_training, args.model_name, model_id, args.predict_neutral_class,
                          args.display_fig, args.label_type, device)

        loss_function_val, pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            evaluate_torch_model(test_list, param_archi, param_training, args.model_name, model_id, all_scores,
                                 args.predict_neutral_class, args.threshold_emo_present, args.label_type,
                                 args.save_predictions, device)

    # Compute and save metrics
    compute_and_save_model_metrics(test_list,
                                   loss_function_val,
                                   pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom,
                                   param_training, args.label_type, args.model_name, model_id, all_scores,
                                   args.predict_neutral_class, args.threshold_emo_present, args.round_decimals,
                                   ext_name, args.plot_confusion_matrix, args.display_fig)


if __name__ == "__main__":
    main()
