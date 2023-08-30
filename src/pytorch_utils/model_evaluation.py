from src.pytorch_utils.dataset_loader import DatasetTorch
from src.label_processing import compute_pred_labels
from src.hyperparam_processing import get_model_module, get_loss_function

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from alive_progress import alive_bar


def evaluate_torch_model(test_list, param_archi, param_training, model_name, model_id, num_steps, all_scores,
                         predict_neutral_class, threshold_emo_pres, label_type, save_pred, device):

    x_test = test_list[0]  # list of lists of 3 arrays
    y_test = test_list[1]
    seg_test = test_list[2]
    num_classes = y_test[0].shape[1]

    num_features_text = x_test[0][0].shape[1]
    num_features_image = x_test[0][1].shape[1]
    num_features_audio = x_test[0][2].shape[1]

    # Load best model
    model_folder = os.path.join('results', model_name)
    model_path = os.path.join(model_folder, 'models', "model_{}.pt".format(model_id))
    model_module = get_model_module(model_name, is_tf=False)
    model = model_module.MModel(model_id, param_archi, num_features_text, num_features_image, num_features_audio,
                                num_classes).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dataloader for test data
    test_data = DatasetTorch(x_test, y_test, seg_test, num_steps, device)
    test_dataloader = DataLoader(test_data)

    test_size = len(test_dataloader.dataset)
    loss_fn = get_loss_function(param_training['loss_function'], is_tf=False)
    test_loss = 0
    # correct = 0
    pred_raw = []

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
    # correct /= test_size
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Avg test loss: {test_loss:>8f} \n")
    pred_raw_list = [pred_raw[i].tolist() for i in range(len(pred_raw))]
    pred_raw_array = np.array(pred_raw_list).reshape(test_size, -1)

    pred_scores_coa, pred_classes_pres, pred_classes_dom = \
        compute_pred_labels(pred_raw_array, all_scores, predict_neutral_class, threshold_emo_pres, save_pred, model_id,
                            num_classes, model_folder, label_type)

    return test_loss, pred_raw_array, pred_scores_coa, pred_classes_pres, pred_classes_dom
