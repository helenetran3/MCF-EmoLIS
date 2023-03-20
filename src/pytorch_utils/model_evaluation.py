from src.utils.dataset_loader import DatasetTorch

import os
import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_torch_model(test_list):

    # Load best model
    model_folder = os.path.join('results', model_name)
    model_path = os.path.join(model_folder, '../models', "model_{}.h5".format(model_id))
    # model = load_model(model_path)  #TODO load model

    x_test = test_list[0]  # list of lists of 3 arrays
    y_test = test_list[1]
    seg_test = test_list[2]

    # Use either gpu or cpu device for evaluation #TODO add in the main code
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Create dataloader for test data
    test_data = DatasetTorch(x_test, y_test, seg_test, device)
    test_dataloader = DataLoader(test_data)

    test_size = len(test_dataloader.dataset)
    loss_fn = nn.L1Loss()  # TODO

    model.eval()  #TODO re-check the code
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            if device == 'cpu':
                X = X.bfloat16()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # TODO formule à vérifier
    test_loss /= test_size # TODO verify if per batch or not
    correct /= test_size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    test_loss, pred_raw, pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom = None, None, None, None, None
    return test_loss, pred_raw, pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom