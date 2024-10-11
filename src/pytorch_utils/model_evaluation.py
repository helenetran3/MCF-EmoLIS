from src.pytorch_utils.dataset_loader import DatasetTorch
from src.label_processing import compute_pred_labels, from_one_hot_to_plain
from src.hyperparam_processing import get_model_module, get_loss_function, get_val_metrics
from src.model_metrics import compute_best_thres_for_monitor_metric

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from alive_progress import alive_bar


def run_one_evaluation(test_list, with_cross_valid, fold_id, param_task, param_model, param_log, num_steps,
                       model_name, model_id, loss_fn_str, val_met_str, thres_multilabel, all_scores, device):

    predict_neutral_class = param_task['predict_neutral_class']
    thres_score_pres = [param_task['thres_score_pres']]
    label_type = param_task['label_type']
    one_hot = ('oh' in label_type)
    round_decimals = param_log['round_decimals']
    save_pred = param_log['save_predictions']
    ext_model_name = "{}_fold_{}".format(model_id, fold_id) if with_cross_valid else model_id

    x_test = test_list[0]  # list of lists of 3 arrays
    y_test = test_list[1]
    y_test_array = np.vstack(y_test)
    seg_test = test_list[2]
    # In the following, predict neutral class is already inferred from label preprocessing
    num_classes = y_test[0].shape[1] if not one_hot else y_test[0].shape[0]

    num_features_text = x_test[0][0].shape[1]
    num_features_image = x_test[0][1].shape[1]
    num_features_audio = x_test[0][2].shape[1]

    # Create dataloader for test data
    test_data = DatasetTorch(x_test, y_test, seg_test, num_steps, one_hot, device)
    test_dataloader = DataLoader(test_data)

    test_size = len(test_dataloader.dataset)
    loss_fn = get_loss_function(loss_fn_str, is_tf=False)
    val_metrics = get_val_metrics(val_met_str, is_tf=False)
    loss_fn_value, val_met_value = 0, 0
    pred_raw = []

    # Load best model
    model_folder = os.path.join('results', model_name)
    model_path = os.path.join(model_folder, 'models', "model_{}.pt".format(ext_model_name))
    model_module = get_model_module(model_name, is_tf=False)
    model = model_module.MModel(model_id, param_model, num_features_text, num_features_image, num_features_audio,
                                num_classes).to(device)
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
                        y = y.bfloat16()

                else:  # Fixed sequence length
                    x_text, x_audio, x_image = X
                    if device == 'cpu':
                        x_text = x_text.float()
                        x_audio = x_audio.float()
                        x_image = x_image.float()
                        y = y.float()

                pred = model(x_text, x_audio, x_image)
                pred_raw += pred.tolist()
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

    # This block creates pred_raw_array, i.e., pred_raw in array format, with shape (test_size, num_classes) and
    # values in [0, 3]
    pred_shape = (test_size, -1, 4) if one_hot else (test_size, -1)  # 4 for number of scores
    pred_raw_array = np.array(pred_raw).reshape(pred_shape)
    if 'oh' in label_type:
        pred_raw_array = from_one_hot_to_plain(pred_raw_array, num_classes)
    if 'prob' in label_type:
        pred_raw_array = pred_raw_array*3

    # This block computes the validation metrics for the test set
    if val_metrics is None:
        _, test_monitor = compute_best_thres_for_monitor_metric(y_test_array, pred_raw_array, round_decimals,
                                                                thres_multilabel)
        test_monitor_avg = np.round(np.mean(test_monitor), round_decimals)
        val_met_value = test_monitor_avg
        print(f"Avg test F1 score: {test_monitor_avg}; per emo: {test_monitor}")
        print(f"Threshold per emo (calculated from validation set if thres_ml_from_best_metric set to 1):",
              f"{thres_multilabel}")
    else:
        print(f"Avg test valid. metric ({val_met_str}): {val_met_value:>8f}")

    pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
        compute_pred_labels(pred_raw_array, all_scores, predict_neutral_class, thres_score_pres, thres_multilabel,
                            save_pred, model_id, num_classes, model_folder, label_type)

    return [loss_fn_value, val_met_value], pred_raw_array, pred_scores_all, pred_scores_coa, pred_classes_pres, \
        pred_classes_dom


def evaluate_torch_model(test_list, param_task, param_model, param_training, param_log, num_steps, model_name, model_id,
                         loss_function, val_metrics, thres_multilabel, all_scores, device, with_cross_valid):

    if with_cross_valid:
        num_folds = param_training['cross_val_folds']
        model_eval_output, pred_raw_array, pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            [], [], [], [], [], []
        for fold_i in range(num_folds):
            loss_vals_i, pred_raw_i, pred_all_i, pred_coa_i, pred_pres_i, pred_dom_i = \
                run_one_evaluation(test_list, with_cross_valid, fold_i,
                                   param_task, param_model, param_log, num_steps, model_name, model_id,
                                   loss_function, val_metrics, thres_multilabel, all_scores, device)
            model_eval_output += [loss_vals_i]  # loss_vals_i = [loss_fn_value, val_met_value]
            pred_raw_array += [pred_raw_i]
            pred_scores_all += [pred_all_i]
            pred_scores_coa += [pred_coa_i]
            pred_classes_pres += [pred_pres_i]
            pred_classes_dom += [pred_dom_i]
    else:
        model_eval_output, pred_raw_array, pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            run_one_evaluation(test_list, with_cross_valid, None,
                               param_task, param_model, param_log, num_steps, model_name, model_id,
                               loss_function, val_metrics, thres_multilabel, all_scores, device)

    return model_eval_output, pred_raw_array, pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom
