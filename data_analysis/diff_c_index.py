import torch
import numpy as np
import csv, os

from src.pickle_functions import pickle_file_exists, load_from_pickle
from src.dataset_utils import get_dataset_from_sdk, get_fold_ids, process_and_split_dataset, get_length_dataset, \
    split_for_cross_validation, data_normalization
from src.hyperparam_processing import get_hyperparam_from_json
from src.label_processing import create_label_filename, update_labels
from src.pytorch_utils.model_evaluation import evaluate_torch_model
from src.csv_functions import create_csv_path

from alive_progress import alive_bar
from sklearn.utils import resample
from src.model_metrics import concordance_index_1d

def compute_c_index_diff_bootstrap(test_list_task, num_folds, pred_raw_s, pred_raw_mts, label_type_reg, all_scores,
                                   num_bootstrap=5000, ci=95):
    """
    Return a dictionary containing the mean of c-indexes, the lower and upper limits of the confidence interval, and
    all c-index values obtained from the bootstrap. The first two parameters can be a list of arrays when there is
    cross-validation.

    :param true_scores_all: array of shape (test size, 6) giving the true scores for the 6 emotions (given by the database).
    :param pred_raw: array of shape (test_size, 6) predicting the presence score of the 6 emotions
    :param label_type: Type of labels used for model training and evaluation
    :param all_scores: list of available presence scores
    :param num_bootstrap: Number of bootstrap resamplings (default 1000)
    :param ci: Confidence interval in percentage (default 95)
    """

    y_test = [test_list_task[fold_i][1] for fold_i in range(num_folds)]
    true_scores_all = np.reshape(np.array(y_test), (-1, 6))

    # Stack arrays from different folds if needed
    if type(pred_raw_s) == list:
        pred_raw_s = np.vstack(pred_raw_s)
        pred_raw_mts = np.vstack([d['fine_task'] for d in pred_raw_mts])

    num_samples, num_classes = true_scores_all.shape[0], true_scores_all.shape[1]

    # Bootstrap resampling
    indices = np.arange(num_samples)
    c_ind_list_s, c_ind_list_mts, c_ind_list_diff = [], [], []
    c_ind_macro_s, c_ind_wei_s = [], []
    c_ind_macro_mts, c_ind_wei_mts = [], []
    c_ind_macro_diff, c_ind_wei_diff = [], []
    weights_emo = list(np.sum(true_scores_all > 0, axis=0))

    with alive_bar(num_bootstrap, title='Bootstrap resampling for C-index') as bar:
        for n_i in range(num_bootstrap):
            sample_idx = resample(indices, replace=True, n_samples=num_samples)
            c_ind_per_emo_s = []
            c_ind_per_emo_mts = []
            c_ind_per_emo_diff = []

            for cl_i in range(num_classes):
                c_ind_s = concordance_index_1d(true_scores_all[sample_idx, cl_i],
                                               pred_raw_s[sample_idx, cl_i], label_type_reg, all_scores)
                c_ind_mts = concordance_index_1d(true_scores_all[sample_idx, cl_i],
                                                 pred_raw_mts[sample_idx, cl_i], label_type_reg, all_scores)
                c_ind_diff = c_ind_mts - c_ind_s
                c_ind_per_emo_s.append(c_ind_s)
                c_ind_per_emo_mts.append(c_ind_mts)
                c_ind_per_emo_diff.append(c_ind_diff)

            # Save all C-indexes per emotion in a list
            c_ind_list_s.append(c_ind_per_emo_s)
            c_ind_list_mts.append(c_ind_per_emo_mts)
            c_ind_list_diff.append(c_ind_per_emo_diff)

            # Compute unweighted C-index for both models + their difference
            c_ind_macro_s.append(np.mean(c_ind_per_emo_s))
            c_ind_macro_mts.append(np.mean(c_ind_per_emo_mts))
            c_ind_macro_diff.append(np.mean(c_ind_per_emo_diff))

            # Compute weighted C-index for both models + their difference
            c_ind_wei_s.append(np.average(c_ind_per_emo_s, weights=weights_emo))
            c_ind_wei_mts.append(np.average(c_ind_per_emo_mts, weights=weights_emo))
            c_ind_wei_diff.append(np.average(c_ind_per_emo_diff, weights=weights_emo))
            bar()

    def ci_lower_values(c_array):
        return np.percentile(c_array, (100 - ci) / 2)

    def ci_upper_values(c_array):
        return np.percentile(c_array, 100 - (100 - ci) / 2)

    def get_ic_values(c_ind_list, c_ind_macro, c_ind_wei):
        c_index_arr = np.array(c_ind_list)  # array of shape (num_bootstrap, num_classes)
        c_index_arr_per_emo = [c_index_arr[:, cl_i] for cl_i in range(num_classes)]
        ci_lower_emo = [ci_lower_values(c_arr) for c_arr in c_index_arr_per_emo]
        ci_upper_emo = [ci_upper_values(c_arr) for c_arr in c_index_arr_per_emo]
        ci_lower_macro = ci_lower_values(np.array(c_ind_macro))
        ci_upper_macro = ci_upper_values(np.array(c_ind_macro))
        ci_lower_wei = ci_lower_values(np.array(c_ind_wei))
        ci_upper_wei = ci_upper_values(np.array(c_ind_wei))

        final_values = {
            "c_index_emo": [np.mean(c_arr) for c_arr in c_index_arr_per_emo],
            "c_index_macro": np.mean(c_ind_macro),
            "c_index_weighted": np.mean(c_ind_wei),
            "ci_lower_emo": ci_lower_emo,
            "ci_upper_emo": ci_upper_emo,
            "ci_lower_macro": ci_lower_macro,
            "ci_upper_macro": ci_upper_macro,
            "ci_lower_weighted": ci_lower_wei,
            "ci_upper_weighted": ci_upper_wei
        }
        for i in range(num_classes):
            print(f"Emotion {i + 1:02d}: {final_values['c_index_emo'][i]:.3f} "
                  f"[{final_values['ci_lower_emo'][i]:.3f}, {final_values['ci_upper_emo'][i]:.3f}]")

        print("\nMacro-average: {:.3f} [{:.3f}, {:.3f}]".format(
            final_values["c_index_macro"],
            final_values["ci_lower_macro"],
            final_values["ci_upper_macro"]
        ))

        print("Weighted-average: {:.3f} [{:.3f}, {:.3f}]".format(
            final_values["c_index_weighted"],
            final_values["ci_lower_weighted"],
            final_values["ci_upper_weighted"]
        ))
        print("─────────────────────────────────────────────\n")

        return final_values

    print("=========================================")
    print("Results of bootstrap sampling for C-index")
    print("=========================================")
    print("\n\nMonotask model:")
    final_values_s = get_ic_values(c_ind_list_s, c_ind_macro_s, c_ind_wei_s)
    print("\n\nMultitask model:")
    final_values_mts = get_ic_values(c_ind_list_mts, c_ind_macro_mts, c_ind_wei_mts)
    print("\n\nDifference in performance:")
    final_values_diff = get_ic_values(c_ind_list_diff, c_ind_macro_diff, c_ind_wei_diff)

    return final_values_s, final_values_mts, final_values_diff


def save_ic_values_to_csv(all_final_values, model_s, id_s, model_mts, id_mts, with_cross_val):
    emotions = ['happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear']
    filename = f"diff_ic_{model_s}_{id_s}_{model_mts}_{id_mts}.csv"
    csv_path_ic_val = create_csv_path("summary", filename, with_cross_val=with_cross_val,
                                       is_summary=True)
    # This additional line since we are in "data_analysis" folder
    csv_path_ic_val = os.path.join("../", csv_path_ic_val)

    with open(csv_path_ic_val, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ModelType",
            "Metric",
            "Mean",
            "CI_lower",
            "CI_upper"
        ])

        # Loop on all final_values
        for model_type, final_values in all_final_values.items():
            # Emotion-level values
            for i, (mean_val, ci_l, ci_u) in enumerate(
                zip(final_values["c_index_emo"],
                    final_values["ci_lower_emo"],
                    final_values["ci_upper_emo"])
            ):
                writer.writerow([model_type, emotions[i], mean_val, ci_l, ci_u])

            # Macro-average
            writer.writerow([
                model_type,
                "Macro-average",
                final_values["c_index_macro"],
                final_values["ci_lower_macro"],
                final_values["ci_upper_macro"]
            ])

            # Weighted-average
            writer.writerow([
                model_type,
                "Weighted-average",
                final_values["c_index_weighted"],
                final_values["ci_lower_weighted"],
                final_values["ci_upper_weighted"]
            ])

    print(f"All bootstrap results saved to: {csv_path_ic_val}")


def compute_IC_diff_c_index(model_names, name_id_dict, loss_function_l, val_metrics_l, label_type_per_model,
                            label_type_reg):

    param_log = get_hyperparam_from_json("param_logistics")
    param_training, param_task = get_hyperparam_from_json("param_training_task")
    num_folds = param_training['cross_val_folds']
    with_cross_validation = (num_folds != -1)

    # Get data for training, validation and test sets from pickle (split provided by the SDK)
    root_data_folder = param_log['root_data_folder']
    tnan = param_task['thres_nan']
    nan_str = param_task['replace_nan'] + "_tnan_" + str(tnan)
    thres_score = param_task['thres_score_pres']
    tscore_str = "_tscore_" + str(thres_score)
    length_seq = param_task['set_length']
    length_str = "_length_" + str(length_seq) if type(length_seq) == int else ""
    data_ext_name_raw = nan_str + length_str
    data_ext_name_proc = nan_str + tscore_str + length_str

    predict_neutral_class = param_task['predict_neutral_class']

    def load_corresponding_data(label_type_j):
        label_filename = create_label_filename(label_type_j)  # for multitask, it looks like coa&pres
        cv_filename = "cv_" + str(num_folds) + "_true_" + label_filename + "_" + data_ext_name_proc
        cv_filename += "_pred-neu" if predict_neutral_class else "_rm-neu"

        # Load folds for cross-validation if the targeted task (mono or multi) are already saved locally
        if with_cross_validation and pickle_file_exists("train_" + cv_filename, "processed_folds", root_data_folder):
            train_list_task = load_from_pickle("train_" + cv_filename, 'processed_folds', root_data_folder)
            valid_list_task = load_from_pickle("valid_" + cv_filename, 'processed_folds', root_data_folder)
            test_list_task = load_from_pickle("test_" + cv_filename, 'processed_folds', root_data_folder)
            all_scores = load_from_pickle("all_scores_with_no_split_" + data_ext_name_raw, "raw_folds", root_data_folder)
            print("\nLoaded {} folds for cross-validation, for task {}, max NaN of {}%, max length of {}, score threshold of"
                  " {} and predict neutral class is {}, from folder '{}/processed_folds'."
                  .format(num_folds, label_type_j, tnan, length_seq, thres_score, (predict_neutral_class == 1),
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
                    train_list_orig = load_from_pickle("train_" + data_ext_name_raw, "raw_folds", root_data_folder)
                    valid_list_orig = load_from_pickle("valid_" + data_ext_name_raw, "raw_folds", root_data_folder)
                    test_list_orig = load_from_pickle("test_" + data_ext_name_raw,  "raw_folds", root_data_folder)
                    no_split_list_orig = load_from_pickle("no_split_" + data_ext_name_raw,  "raw_folds", root_data_folder)
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
                                                  with_cross_validation)

                # Update labels so that it matches with the targeted task given by label_type (all, coarse, coarse_prob,
                # coarse_oh, present, dominant) + remove samples assigned to neutral when predict_neutral_class is False
                # In the case of multitask, the second element of each of the four lists are a dictionary with keys
                # 'coarse_task' and 'fine_task'
                train_list_task, valid_list_task, test_list_task, no_split_list_task = \
                    update_labels(train_list_orig, valid_list_orig, test_list_orig, no_split_list_orig, param_task,
                                  label_type_j, all_scores, data_ext_name_proc, root_data_folder)

            # Cross-validation
            if with_cross_validation:
                train_list_task, valid_list_task, test_list_task = \
                    split_for_cross_validation(train_list_orig, valid_list_orig, test_list_orig, no_split_list_orig,
                                               train_list_task, valid_list_task, test_list_task, no_split_list_task,
                                               param_training, param_task, param_log, label_type_j, data_ext_name_proc)

            del train_list_orig, valid_list_orig, test_list_orig, no_split_list_orig

        # Data normalization: we perform it after cross-validation (or hold-out) because it uses values from training set
        train_list_task, valid_list_task, test_list_task = data_normalization(train_list_task, valid_list_task,
                                                                              test_list_task, param_task,
                                                                              with_cross_validation)

        return train_list_task, valid_list_task, test_list_task, all_scores

    def run_one_inference(model_name_j, loss_function_j, val_metrics_j, param_model_j, label_type_model_j):

        train_list_task, valid_list_task, test_list_task, all_scores = load_corresponding_data(label_type_model_j)
        num_steps = get_length_dataset(train_list_task, valid_list_task, test_list_task, param_task,
                                       with_cross_validation, is_tf=False)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\nUsing {device} device")

        param_task['label_type'] = label_type_model_j
        model_id_j = name_id_dict[model_name_j]
        thres_multilabel = [{'coarse_task': [1, 1, 1, 1, 1, 1], 'fine_task': None},
                            {'coarse_task': [1, 1, 1, 1, 1, 1], 'fine_task': None},
                            {'coarse_task': [1, 1, 1, 1, 1, 1], 'fine_task': None},
                            {'coarse_task': [1, 1, 1, 1, 1, 1], 'fine_task': None},
                            {'coarse_task': [1, 1, 1, 1, 1, 1], 'fine_task': None}]

        _, pred_raw, _, _ = \
            evaluate_torch_model(test_list_task, param_task, param_model_j, param_training, param_log, num_steps,
                                 model_name_j, model_id_j, label_type_model_j, loss_function_j, val_metrics_j,
                                 thres_multilabel, device, with_cross_validation)

        return pred_raw, test_list_task, all_scores

    # Compute predictions of already saved models
    pred_raw_s, pred_raw_mts = None, None
    for model_i, lf_i, vm_i, lt_i in zip(model_names, loss_function_l, val_metrics_l, label_type_per_model):
        param_model_i = get_hyperparam_from_json("param_model", model_name=model_i)
        pred_model, test_list_task, all_scores = run_one_inference(model_i, lf_i, vm_i, param_model_i, lt_i)
        if type(lt_i) == dict:
            pred_raw_mts = pred_model
        else:
            pred_raw_s = pred_model

    values_s, values_mts, values_diff = compute_c_index_diff_bootstrap(test_list_task, num_folds,
                                                                       pred_raw_s, pred_raw_mts,
                                                                       label_type_reg, all_scores)

    all_final_values = {
        "monotask": values_s,
        "multitask": values_mts,
        "difference": values_diff
    }

    model_s_name, model_mts_name = model_names[1], model_names[0]
    id_s = name_id_dict[model_s_name]
    id_mts = name_id_dict[model_mts_name]

    save_ic_values_to_csv(all_final_values, model_s_name, id_s, model_mts_name, id_mts,
                          with_cross_val=(num_folds != -1))
    

# Place the multitask model on first position of the list, to avoid issues
model_name_list = ['mult', 'lf_trans_me2e']
model_name_id = {'mult': 18, 'lf_trans_me2e': 6}
loss_function_list = [{'coarse_task': 'wei_bce', 'fine_task': 'mse'}, 'mse']
val_metrics_list = [{'coarse_task': 'wei_bce', 'fine_task': 'mse'}, 'mse']
label_type_per_model = [{'coarse_task': 'present', 'fine_task': 'all'}, 'all']
label_type_reg = 'all'
compute_IC_diff_c_index(model_name_list, model_name_id, loss_function_list, val_metrics_list,
                        label_type_per_model, label_type_reg)
