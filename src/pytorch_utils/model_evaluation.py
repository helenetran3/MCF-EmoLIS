from src.pytorch_utils.dataset_loader import DatasetTorch
from src.label_processing import compute_pred_labels, from_one_hot_to_plain
from src.hyperparam_processing import get_model_module, get_loss_function, get_val_metrics
from src.model_metrics import compute_best_thres_for_monitor_metric

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from alive_progress import alive_bar


def run_one_evaluation(test_list, with_cross_valid, fold_id, param_task, param_model, param_training, param_log,
                       num_steps, model_name, model_id, label_type, loss_fn_str, val_met_str, thres_multilabel,
                       all_scores, device):

    predict_neutral_class = param_task['predict_neutral_class']
    thres_score_pres = [param_task['thres_score_pres']]
    one_hot = ("oh" in label_type) if type(label_type) != dict else \
        {'coarse_task': "oh" in label_type['coarse_task'], 'fine_task': "oh" in label_type['fine_task']}
    round_decimals = param_log['round_decimals']
    save_pred = param_log['save_predictions']
    ext_model_name = "{}_fold_{}".format(model_id, fold_id) if with_cross_valid else model_id

    x_test = test_list[0]  # list of lists of 3 arrays
    y_test = test_list[1]
    y_test_array = np.vstack(y_test) if type(y_test) != dict else \
        {'coarse_task': np.vstack(y_test['coarse_task']), 'fine_task': np.vstack(y_test['fine_task'])}
    seg_test = test_list[2]
    # In the following, predict neutral class is already inferred from label preprocessing
    if type(label_type) == dict:
        num_classes = y_test['fine_task'][0].shape[1] if not one_hot['fine_task'] else y_test['fine_task'][0].shape[0]
    else:
        num_classes = y_test[0].shape[1] if not one_hot else y_test[0].shape[0]

    num_features_text = x_test[0][0].shape[1]
    num_features_image = x_test[0][1].shape[1]
    num_features_audio = x_test[0][2].shape[1]

    # Create dataloader for test data
    test_data = DatasetTorch(x_test, y_test, seg_test, num_steps, one_hot, device)
    test_dataloader = DataLoader(test_data)

    test_size = len(test_dataloader.dataset)
    loss_fn = get_loss_function(loss_fn_str, param_training, is_tf=False)
    val_metrics = get_val_metrics(val_met_str, param_training, is_tf=False)
    loss_fn_value, val_met_value = 0, 0
    pred_raw = {'coarse_task': [], 'fine_task': []} if type(label_type) == dict else []

    # Load best model
    model_folder = os.path.join('results', model_name)
    model_path = os.path.join(model_folder, 'models', "model_{}.pt".format(ext_model_name))
    model_module = get_model_module(model_name, is_tf=False)
    model = model_module.MModel(model_id, param_model, param_task, num_features_text, num_features_image,
                                num_features_audio, num_classes).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    fold_text = ": Fold {}".format(fold_id) if with_cross_valid else ""
    print("\n\n================================= Model Evaluation{} =============================================\n"
          .format(fold_text))

    with torch.no_grad():
        with alive_bar(test_size, title='Model evaluation') as bar:
            for X, y in test_dataloader:

                if num_steps is None:  # Varying sequence length
                    # X, y = X.to(device), y.to(device)
                    x_text, x_audio, x_image = X
                    # [0] to remove the 1st dimension (initially shape: (1, length, num_feat)
                    x_text, x_audio, x_image, y = x_text[0], x_audio[0], x_image[0], y[0]
                    if device == 'cpu':
                        x_text = x_text.bfloat16()
                        x_audio = x_audio.bfloat16()
                        x_image = x_image.bfloat16()
                        if type(y) == dict:
                            for task_i in ['coarse_task', 'fine_task']:
                                y[task_i] = y[task_i].bfloat16()
                        else:
                            y = y.bfloat16()

                else:  # Fixed sequence length
                    x_text, x_audio, x_image = X
                    if device == 'cpu':
                        x_text = x_text.float()
                        x_audio = x_audio.float()
                        x_image = x_image.float()
                        if type(y) == dict:
                            for task_i in ['coarse_task', 'fine_task']:
                                y[task_i] = y[task_i].float()
                        else:
                            y = y.float()

                pred = model(x_text, x_audio, x_image)
                if type(pred) == dict:
                    pred_raw['coarse_task'] += pred['coarse_task'].tolist()
                    pred_raw['fine_task'] += pred['fine_task'].tolist()
                loss_fn_value += loss_fn(pred, y).item()
                if val_metrics is not None:
                    val_met_value += val_metrics(pred, y).item()

                bar()
    loss_fn_value /= test_size
    loss_fn_value = np.round(loss_fn_value, round_decimals)
    if val_metrics is not None:
        val_met_value /= test_size
        val_met_value = np.round(val_met_value, round_decimals)

    print(f"Avg test loss ({loss_fn_str}): {loss_fn_value:>8f}")

    # # Only for debugging
    # from src.pickle_functions import load_from_pickle
    # model_name = "mult"
    # model_id = 2
    # pred_raw = load_from_pickle("pred_raw_"+str(model_id), "predictions", "results/"+model_name)

    ####################################################################################################################
    # This block creates pred_raw_array, i.e., pred_raw in array format, with shape (test_size, num_classes) and
    # values in [0, 3]
    ####################################################################################################################
    def pred_raw_post_proc(pred_r, label_t, oh):
        p_shape = (test_size, -1, 4) if oh else (test_size, -1) # 4 for number of scores
        out = np.array(pred_r).reshape(p_shape)
        if 'oh' in label_t:
            out = from_one_hot_to_plain(out, num_classes)
        if 'prob' in label_t:
            out = out*3
        return out

    if type(pred_raw) == dict:
        pred_raw_array = {}
        for task_i in ['coarse_task', 'fine_task']:
            pred_raw_array[task_i] = pred_raw_post_proc(pred_raw[task_i], label_type[task_i], one_hot[task_i])
    else:
        pred_raw_array = pred_raw_post_proc(pred_raw, label_type, one_hot)

    ####################################################################################################################
    # This block computes the validation metrics for the test set
    ####################################################################################################################
    def test_validation_metrics(y_test_arr, pred_raw_arr, thres_ml_list):
        _, test_monitor = compute_best_thres_for_monitor_metric(y_test_arr, pred_raw_arr, round_decimals, thres_ml_list)
        test_monitor_avg = np.round(np.mean(test_monitor), round_decimals)
        print(f"Avg test F1 score: {test_monitor_avg}; per emo: {test_monitor}")
        print(f"Threshold per emo (calculated from validation set if thres_ml_from_best_metric set to 1):",
              f"{thres_multilabel}")
        return test_monitor_avg

    if val_metrics is None:
        if type(pred_raw_array) == dict:
            val_met_value = []
            for task_i in ['coarse_task', 'fine_task']:
                print(f"----- Results for task '{label_type[task_i]}' -----")
                test_met_avg = test_validation_metrics(y_test_array[task_i], pred_raw_array[task_i],
                                                       thres_multilabel[task_i])
                val_met_value += [test_met_avg]
        else:
            val_met_value = test_validation_metrics(y_test_array, pred_raw_array, thres_multilabel)
    else:
        print(f"Avg test valid. metric ({val_met_str}): {val_met_value:>8f}")

    ####################################################################################################################
    # This block computes the prediction vectors for the four tasks 'all', 'coarse', 'present', 'dominant"
    ####################################################################################################################
    def compute_pred_label_i(pred_raw_arr, thres_ml_list, label_t, text_id, multitask):
        pred_all, pred_coa, pred_pres, pred_dom = \
            compute_pred_labels(pred_raw_arr, all_scores, predict_neutral_class, thres_score_pres,
                                thres_ml_list, save_pred, text_id, num_classes, model_folder, label_t, multitask)
        return pred_all, pred_coa, pred_pres, pred_dom

    if type(label_type) == dict:
        pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom = None, None, None, None
        for task_i in ['coarse_task', 'fine_task']:
            pred_all_i, pred_coa_i, pred_pres_i, pred_dom_i = compute_pred_label_i(pred_raw_array[task_i],
                                                                                   thres_multilabel[task_i],
                                                                                   label_type[task_i],
                                                                                   str(model_id) + task_i[:3],
                                                                                   multitask=True)
            pred_scores_all = pred_all_i if pred_all_i is not None else pred_scores_all
            pred_scores_coa = pred_coa_i if pred_coa_i is not None else pred_scores_coa
            pred_classes_pres = pred_pres_i if pred_pres_i is not None else pred_classes_pres
            pred_classes_dom = pred_dom_i if pred_dom_i is not None else pred_classes_dom
    else:
        pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            compute_pred_label_i(pred_raw_array, thres_multilabel, label_type, model_id, multitask=False)

    return [loss_fn_value, val_met_value], pred_raw_array, pred_scores_all, pred_scores_coa, pred_classes_pres, \
        pred_classes_dom


def evaluate_torch_model(test_list, param_task, param_model, param_training, param_log, num_steps, model_name, model_id,
                         label_type, loss_function, val_metrics, thres_multilabel, all_scores, device, with_cross_valid):

    if with_cross_valid:
        num_folds = param_training['cross_val_folds']
        model_eval_output, pred_raw_array, pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            [], [], [], [], [], []
        for fold_i in range(num_folds):
            loss_vals_i, pred_raw_i, pred_all_i, pred_coa_i, pred_pres_i, pred_dom_i = \
                run_one_evaluation(test_list[fold_i], with_cross_valid, fold_i,
                                   param_task, param_model, param_training, param_log, num_steps, model_name, model_id,
                                   label_type, loss_function, val_metrics, thres_multilabel[fold_i], all_scores, device)
            model_eval_output += [loss_vals_i]  # loss_vals_i = [loss_fn_value, val_met_value]
            pred_raw_array += [pred_raw_i]
            pred_scores_all += [pred_all_i]
            pred_scores_coa += [pred_coa_i]
            pred_classes_pres += [pred_pres_i]
            pred_classes_dom += [pred_dom_i]
    else:
        model_eval_output, pred_raw_array, pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            run_one_evaluation(test_list, with_cross_valid, None,
                               param_task, param_model, param_training, param_log, num_steps, model_name, model_id,
                               label_type, loss_function, val_metrics, thres_multilabel, all_scores, device)

    return model_eval_output, pred_raw_array, pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom
