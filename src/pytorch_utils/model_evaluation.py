from src.pytorch_utils.dataset_loader import DatasetTorch
from src.models.ef_williams_torch import Model_ef
from src.label_processing import compute_pred_labels

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_torch_model(test_list, num_layers, num_nodes, dropout_rate, final_activ, model_name, model_id, all_scores,
                         predict_neutral_class, threshold_emo_pres, save_pred, device):

    x_test = test_list[0]  # list of lists of 3 arrays
    y_test = test_list[1]
    seg_test = test_list[2]
    num_classes = y_test[0].shape[1]

    # TODO check values
    num_features_text = x_test[0][0].shape[1]
    num_features_image = x_test[0][1].shape[1]
    num_features_audio = x_test[0][2].shape[1]

    # Load best model
    model_folder = os.path.join('results', model_name)
    model_path = os.path.join(model_folder, '../models', "model_{}.pt".format(model_id))
    # TODO check load model
    model = Model_ef(model_id, num_features_text, num_features_image, num_features_audio, num_classes, num_layers,
                     num_nodes, dropout_rate, final_activ).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dataloader for test data
    test_data = DatasetTorch(x_test, y_test, seg_test, device)
    test_dataloader = DataLoader(test_data)

    test_size = len(test_dataloader.dataset)
    loss_fn = nn.L1Loss()  # TODO
    test_loss = 0
    # correct = 0
    pred_raw = []

    # TODO re-check the code
    with torch.no_grad():
        for X, y in test_dataloader:
            # X, y = X.to(device), y.to(device)
            X_text, X_audio, X_image = X
            # [0] to remove the 1st dimension (initially shape: (1, length, num_feat)
            X_text, X_audio, X_image, y = X_text[0], X_audio[0], X_image[0], y[0]
            if device == 'cpu':
                X_text = X_text.bfloat16()
                X_audio = X_audio.bfloat16()
                X_image = X_image.bfloat16()
                y = y.bfloat16()
            pred = model(X_text, X_audio, X_image)
            # TODO check this following line
            pred_raw += pred
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # TODO Accuracy, might not need this
    test_loss /= test_size  # TODO verify if per batch or not
    # correct /= test_size
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Avg test loss: {test_loss:>8f} \n")
    pred_raw_list = [pred_raw[i].tolist() for i in range(len(pred_raw))]
    pred_raw_array = np.array(pred_raw_list).reshape(test_size, -1)
    # print("pred_raw_array.shape: ", pred_raw_array.shape)

    # TODO check this line
    pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
        compute_pred_labels(pred_raw_array, all_scores, predict_neutral_class, threshold_emo_pres, save_pred, model_id,
                            num_classes, model_folder)

    return test_loss, pred_raw, pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom
