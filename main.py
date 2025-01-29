from src.pickle_functions import pickle_file_exists, load_from_pickle
from src.dataset_utils import get_dataset_from_sdk, get_fold_ids, process_and_split_dataset, get_length_dataset, \
    split_for_cross_validation, data_normalization
from src.hyperparam_processing import get_hyperparam_from_json
from src.csv_functions import get_current_model_id, save_model_param_in_csv_file, csv_summary
from src.label_processing import create_label_multitask, create_label_filename, update_labels
from src.model_metrics import compute_and_save_model_metrics
from src.hyperparam_processing import possible_val_metrics
import itertools


def main():

    param_log = get_hyperparam_from_json("param_logistics")
    param_training, param_task = get_hyperparam_from_json("param_training_task")
    num_folds = param_training['cross_val_folds']
    with_cross_validation = (num_folds != -1)

    # Transform label_type to dict with keys 'coarse_task' and 'fine_task' when label_type is a list
    label_type = param_task['label_type']
    label_type = create_label_multitask(label_type) if type(label_type) == list else label_type

    # Get data for training, validation and test sets from pickle (split provided by the SDK)
    root_data_folder = param_log['root_data_folder']
    tnan = param_task['thres_nan']
    nan_str = param_task['replace_nan'] + "_tnan_" + str(tnan)
    tscore_str = "_tscore_" + str(param_task['thres_score_pres'])
    length_seq = param_task['set_length']
    length_str = "_length_" + str(length_seq) if type(length_seq) == int else ""
    data_ext_name_raw = nan_str + length_str
    data_ext_name_proc = nan_str + tscore_str + length_str

    predict_neutral_class = param_task['predict_neutral_class']
    label_filename = create_label_filename(label_type)  # for multitask, it looks like coa&pres
    cv_filename = "cv_" + str(num_folds) + "_true_" + label_filename + "_" + data_ext_name_proc
    cv_filename += "_pred-neu" if predict_neutral_class else "_rm-neu"

    # Load folds for cross-validation if the targeted task (mono or multi) are already saved locally
    if with_cross_validation and pickle_file_exists("train_" + cv_filename, "processed_folds", root_data_folder):
        train_list_task = load_from_pickle("train_" + cv_filename, 'processed_folds', root_data_folder)
        valid_list_task = load_from_pickle("valid_" + cv_filename, 'processed_folds', root_data_folder)
        test_list_task = load_from_pickle("test_" + cv_filename, 'processed_folds', root_data_folder)
        all_scores = load_from_pickle("all_scores_with_no_split_" + data_ext_name_raw, "raw_folds", root_data_folder)
        print("\nLoaded {} folds for cross-validation, for task {}, max NaN of {}%, max length of {} and predict "
              "neutral class is {}, from folder '{}/processed_folds'.".format(num_folds, label_type, tnan, length_seq,
                                                                              (predict_neutral_class == 1),
                                                                              root_data_folder))

    else:
        # Load hold-out data if the targeted task (mono or multi) are already saved locally
        ho_filename = "ho_true_" + label_filename + "_" + data_ext_name_proc
        ho_filename += "_pred-neu" if predict_neutral_class else "_rm-neu"
        if pickle_file_exists("train_" + ho_filename, "processed_folds", root_data_folder):
            train_list_task = load_from_pickle("train_" + ho_filename, "processed_folds", root_data_folder)
            valid_list_task = load_from_pickle("valid_" + ho_filename, "processed_folds", root_data_folder)
            test_list_task = load_from_pickle("test_" + ho_filename, "processed_folds", root_data_folder)
            if with_cross_validation:
                no_split_list_task = load_from_pickle("no_split_" + ho_filename, "processed_folds", root_data_folder)
            split_str = "with_no_split_" if with_cross_validation else "split_only_"
            all_scores = load_from_pickle("all_scores_" + split_str + data_ext_name_raw, "raw_folds", root_data_folder)

        else:
            # Load raw data from CMU-MOSEI SDK if already saved locally
            if pickle_file_exists("train_" + data_ext_name_raw, "raw_folds", root_data_folder):
                train_list_orig = load_from_pickle("train_" + data_ext_name_raw, "raw_folds", root_data_folder)
                valid_list_orig = load_from_pickle("valid_" + data_ext_name_raw, "raw_folds", root_data_folder)
                test_list_orig = load_from_pickle("test_" + data_ext_name_raw,  "raw_folds", root_data_folder)
                no_split_list_orig = load_from_pickle("no_split_" + data_ext_name_raw,  "raw_folds", root_data_folder)
                split_str = "with_no_split_" if with_cross_validation else "split_only_"
                all_scores = load_from_pickle("all_scores_" + split_str + data_ext_name_raw, "raw_folds", root_data_folder)

            else:
                # Load CMU-MOSEI dataset
                dataset = get_dataset_from_sdk(param_log, param_task)

                # Get data for training, validation and test sets (split provided by the SDK)
                train_ids, valid_ids, test_ids = get_fold_ids(param_task)
                train_list_orig, valid_list_orig, test_list_orig, no_split_list_orig, all_scores = \
                    process_and_split_dataset(dataset, train_ids, valid_ids, test_ids, param_task, param_log,
                                              with_cross_validation, data_ext_name_raw)

            # Update labels so that it matches with the targeted task given by label_type (all, coarse, coarse_prob,
            # coarse_oh, present, dominant) + remove samples assigned to neutral when predict_neutral_class is False
            # In the case of multitask, the second element of each of the four lists are a dictionary with keys
            # 'coarse_task' and 'fine_task'
            train_list_task, valid_list_task, test_list_task, no_split_list_task = \
                update_labels(train_list_orig, valid_list_orig, test_list_orig, no_split_list_orig, param_task,
                              label_type, all_scores, data_ext_name_proc, root_data_folder)

        # Cross-validation
        if with_cross_validation:
            train_list_task, valid_list_task, test_list_task = \
                split_for_cross_validation(train_list_task, valid_list_task, test_list_task, no_split_list_task,
                                           param_training, param_task, param_log, label_type, data_ext_name_proc)

    # Data normalization: we perform it after cross-validation (or hold-out) because it uses values from training set
    train_list_task, valid_list_task, test_list_task = data_normalization(train_list_task, valid_list_task,
                                                                          test_list_task, param_task,
                                                                          with_cross_validation)

    def run_one_experiment(model_name, loss_function, val_metrics, param_model_j):
        model_id = get_current_model_id(model_name)
        is_tf = bool(param_model_j['is_TensorFlow'])
        num_steps = get_length_dataset(train_list_task, valid_list_task, test_list_task, param_task,
                                       with_cross_validation, is_tf)

        # Model training
        if is_tf:
            from src.tensorflow_utils.model_training import train_tf_model
            from src.tensorflow_utils.model_evaluation import evaluate_tf_model

            thres_multilabel = train_tf_model(train_list_task, valid_list_task, test_list_task, param_task,
                                              param_model_j, param_training, param_log, num_steps, model_name, model_id,
                                              loss_function, val_metrics, label_type, with_cross_validation)

            model_eval_output, pred_raw, pred_classes_pres, pred_classes_dom = \
                evaluate_tf_model(test_list_task, param_task, param_training, param_log, num_steps, model_name, model_id,
                                  loss_function, val_metrics, thres_multilabel, with_cross_validation)

        else:
            import torch
            from src.pytorch_utils.model_training import train_torch_model
            from src.pytorch_utils.model_evaluation import evaluate_torch_model

            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"\nUsing {device} device")

            thres_multilabel = train_torch_model(train_list_task, valid_list_task, test_list_task,
                                                 param_task, param_model_j, param_training, param_log,
                                                 num_steps, model_name, model_id, loss_function, val_metrics,
                                                 label_type, device, with_cross_validation)

            model_eval_output, pred_raw, pred_classes_pres, pred_classes_dom = \
                evaluate_torch_model(test_list_task, param_task, param_model_j, param_training,
                                     param_log, num_steps, model_name, model_id, label_type, loss_function, val_metrics,
                                     thres_multilabel, device, with_cross_validation)

        # Save model parameters in csv file
        save_model_param_in_csv_file(model_id, param_task, param_model_j, param_training, model_name,
                                     loss_function, val_metrics, with_cross_validation)

        # Compute and save metrics
        compute_and_save_model_metrics(test_list_task, with_cross_validation, model_eval_output,
                                       pred_raw, pred_classes_pres, pred_classes_dom,
                                       param_task, param_training, param_log, model_name, model_id, label_type,
                                       loss_function, val_metrics, all_scores)

    # Loop over the list of models, loss functions and validation metrics
    model = param_task['model_name']
    loss_function_str = param_training['loss_function']
    val_metrics_str = param_training['val_metrics']
    loss_function_list = loss_function_str if type(loss_function_str) == list else [loss_function_str]
    val_metrics_list = val_metrics_str if type(val_metrics_str) == list else [val_metrics_str]
    for val_met in val_metrics_list:
        if val_met == -1 and ((type(label_type) != list and label_type not in ['present', 'dominant']) or
                              (type(label_type) == list and
                               any([label_i not in ['present', 'dominant'] for label_i in label_type]))):
            raise ValueError("The value '-1' assigned to hyperparameter 'val_metrics' is incompatible with",
                             "tasks" if type(label_type) == list else "task",
                             "'{}'. Please choose a value among {} in 'param_training_task.json' file."
                             .format(param_task['label_type'], possible_val_metrics))

    if type(model) == list:
        for model_i in model:
            param_model_i = get_hyperparam_from_json("param_model", model_name=model_i)
            # Iterate over all pairs (loss_function, val_metrics)
            for lf_i, vm_i in list(itertools.product(loss_function_list, val_metrics_list)):
                run_one_experiment(model_i, lf_i, vm_i, param_model_i)
    else:
        param_model = get_hyperparam_from_json("param_model", model_name=model)
        if model == 'mult' and param_model['run_model_var']:
            # For mult, automate the change of model hyperparameters
            hyperparam_loop = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1]]
            for dominant_modality_i, with_cross_module_i, lav_as_modality_i in hyperparam_loop:
                param_model['dominant_modality'] = dominant_modality_i
                param_model['with_cross_module'] = with_cross_module_i
                param_model['lav_as_modality'] = lav_as_modality_i
                for lf_i, vm_i in list(itertools.product(loss_function_list, val_metrics_list)):
                    run_one_experiment(model, lf_i, vm_i, param_model)
        else:
            for lf_i, vm_i in list(itertools.product(loss_function_list, val_metrics_list)):
                run_one_experiment(model, lf_i, vm_i, param_model)

    # Summary files
    csv_summary(with_cross_validation, param_log)


if __name__ == "__main__":
    main()
