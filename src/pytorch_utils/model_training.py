from src.pytorch_utils.dataset_loader import DatasetTorch, my_collate_fn
from src.models.ef_williams_torch import Model_ef

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary


def train_torch_model(train_list, valid_list, batch_size, num_epochs, num_layers,
                      num_nodes, dropout_rate, final_activ, learning_rate, optimizer_name, loss_function,
                      val_metric, patience, model_name, predict_neutral_class, model_id, display_fig, label_type):

    x_train = train_list[0]  # list of lists of 3 arrays
    y_train = train_list[1]
    seg_train = train_list[2]
    x_valid = valid_list[0]  # list of lists of 3 arrays
    y_valid = valid_list[1]
    seg_valid = valid_list[2]
    num_classes = y_train[0].shape[1]

    # Use either gpu or cpu device for training #TODO add in the main code
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Create dataloaders for training and validation data
    training_data = DatasetTorch(x_train, y_train, seg_train, device)
    valid_data = DatasetTorch(x_valid, y_valid, seg_valid, device)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=my_collate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=my_collate_fn)

    # Create an instance of the model
    num_features_text = x_train[0][0].shape[1]
    num_features_image = x_train[0][1].shape[1]
    num_features_audio = x_train[0][2].shape[1]
    model = Model_ef(model_id, num_features_text, num_features_image, num_features_audio, num_classes, num_layers,
                     num_nodes, dropout_rate, final_activ).to(device)
    summary(model, [(374, 300), (374, 35), (374, 74)], device='cpu', dtypes=[torch.bfloat16, torch.bfloat16, torch.bfloat16])

    # Optimizing the model parameters
    loss_fn = nn.L1Loss()  #TODO
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  #TODO

    # Training
    train_size = len(train_dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(train_dataloader):
        loss = 0
        for X_sample, y_sample in zip(X, y):
            X_text, X_audio, X_image = X_sample
            if device == 'cpu':
                X_text = X_text.bfloat16()
                X_audio = X_audio.bfloat16()
                X_image = X_image.bfloat16()
            pred = model(X_text, X_audio, X_image)
            loss += loss_fn(pred, y_sample)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch % 10 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f} [{current:>5d}/{train_size:>5d}]")

        # TODO save model
        # TODO callbacks : patience and modelcheckpoint

    return 0
