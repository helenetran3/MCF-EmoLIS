from src.pytorch_utils.dataset_loader import DatasetTorch
from src.label_processing import compute_pred_labels
from src.hyperparam_processing import get_model_module, get_loss_function
from src.pickle_functions import save_with_pickle

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from alive_progress import alive_bar


def run_one_evaluation(test_list, with_cross_valid, fold_id, param_task, param_model, param_training, param_log,
                       num_steps, model_name, model_id, all_scores, device):

    predict_neutral_class = param_task['predict_neutral_class']
    threshold_emo_pres = [param_task['threshold_emo_present']]
    label_type = param_task['label_type']
    round_decimals = param_log['round_decimals']
    save_pred = param_log['save_predictions']
    ext_model_name = "{}_fold_{}".format(model_id, fold_id) if with_cross_valid else model_id

    x_test = test_list[0]  # list of lists of 3 arrays
    y_test = test_list[1]
    seg_test = test_list[2]
    num_classes = y_test[0].shape[1]

    num_features_text = x_test[0][0].shape[1]
    num_features_image = x_test[0][1].shape[1]
    num_features_audio = x_test[0][2].shape[1]

    # Create dataloader for test data
    test_data = DatasetTorch(x_test, y_test, seg_test, num_steps, device)
    test_dataloader = DataLoader(test_data)

    test_size = len(test_dataloader.dataset)
    loss_fn = get_loss_function(param_training['loss_function'], is_tf=False)
    test_loss = 0
    # correct = 0
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
                    pred = model(x_text, x_audio, x_image)
                    pred_raw += pred
                    test_loss += loss_fn(pred, y).item()
                    # correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # TODO Accuracy, might not need this

                else:  # Fixed sequence length
                    x_text, x_audio, x_image = X
                    if device == 'cpu':
                        x_text = x_text.float()
                        x_audio = x_audio.float()
                        x_image = x_image.float()
                    pred = model(x_text, x_audio, x_image)
                    pred_raw += pred
                    test_loss += loss_fn(pred, y).item()

                bar()
    test_loss /= test_size
    test_loss = np.round(test_loss, round_decimals)

    # correct /= test_size
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Avg test loss: {test_loss:>8f} \n")
    pred_raw_list = [pred_raw[i].tolist() for i in range(len(pred_raw))]
    pred_raw_array = np.array(pred_raw_list).reshape(test_size, -1)

    if save_pred:
        save_with_pickle(pred_raw, "pred_raw_{}".format(model_id), pickle_folder="predictions",
                         root_folder=model_folder)

    pred_scores_coa, pred_classes_pres, pred_classes_dom = \
        compute_pred_labels(pred_raw_array, all_scores, predict_neutral_class, threshold_emo_pres, save_pred, model_id,
                            num_classes, model_folder, label_type)

    return test_loss, pred_raw_array, pred_scores_coa, pred_classes_pres, pred_classes_dom


def evaluate_torch_model(test_list, param_task, param_model, param_training, param_log, num_steps,
                         model_name, model_id, all_scores, device, with_cross_valid):

    if with_cross_valid:
        num_folds = param_training['cross_val_folds']
        test_loss, pred_raw_array, pred_scores_coa, pred_classes_pres, pred_classes_dom = [], [], [], [], []
        for fold_i in range(num_folds):
            loss_i, pred_raw_i, pred_coa_i, pred_pres_i, pred_dom_i = \
                run_one_evaluation(test_list, with_cross_valid, fold_i, param_task, param_model, param_training,
                                   param_log, num_steps, model_name, model_id, all_scores, device)
            test_loss += [loss_i]
            pred_raw_array += [pred_raw_i]
            pred_scores_coa += [pred_coa_i]
            pred_classes_pres += [pred_pres_i]
            pred_classes_dom += [pred_dom_i]
    else:
        test_loss, pred_raw_array, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            run_one_evaluation(test_list, with_cross_valid, None, param_task, param_model, param_training, param_log,
                               num_steps, model_name, model_id, all_scores, device)

    return test_loss, pred_raw_array, pred_scores_coa, pred_classes_pres, pred_classes_dom
