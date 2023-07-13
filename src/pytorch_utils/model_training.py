from src.pytorch_utils.dataset_loader import DatasetTorch, my_collate_fn
from src.models.ef_williams_torch import Model_ef

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from alive_progress import alive_bar

from matplotlib import pyplot as plt


def model_checkpoint(valid_loss, valid_loss_best, best_is_smallest, model, optimizer, epoch, train_loss,
                     model_name, model_id):

    model_folder = os.path.join('results', model_name, '../models')

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if epoch == 0 \
            or (best_is_smallest and valid_loss < valid_loss_best) \
            or (not best_is_smallest and valid_loss > valid_loss_best):

        valid_loss_best = valid_loss

        # TODO check save model
        model_path = os.path.join(model_folder, "model_{}.pt".format(model_id))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, model_path)

    return valid_loss_best


def model_early_stopping(valid_loss, valid_loss_best, best_is_smallest, num_epochs_without_improvement):
    if (best_is_smallest and valid_loss >= valid_loss_best) or (not best_is_smallest and valid_loss <= valid_loss_best):
        num_epochs_without_improvement += 1
    else:
        num_epochs_without_improvement = 0
    return num_epochs_without_improvement


def plot_history(history_train, history_valid, model_id, val_metric, history_folder, root_folder, display_fig):
    """
    Plot history, display and save figures

    :param history_train: List of training losses
    :param history_valid: List of validation losses
    :param model_id: Model id (int)
    :param val_metric: Metric used to monitor validation data
    :param history_folder: Name of the folder where the history plot will be saved
    :param root_folder: Name of the root folder where the history folder is created
    :param display_fig: Whether we display the figures
    """

    # Create full path folder
    full_path_folder = os.path.join(root_folder, history_folder)
    if not os.path.isdir(full_path_folder):
        os.makedirs(full_path_folder)

    # summarize history for training and validation metric used to monitor
    plt.plot(history_train)
    plt.plot(history_valid)
    plt.title('Model {}'.format(val_metric))
    plt.ylabel(val_metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    history_plot_path = os.path.join(full_path_folder, 'history_{}_{}.png'.format(model_id, val_metric))
    plt.savefig(history_plot_path, bbox_inches='tight')
    if display_fig:
        plt.show()


def train_torch_model(train_list, valid_list, test_list, batch_size, num_epochs, num_layers,
                      num_nodes, dropout_rate, final_activ, learning_rate, optimizer_name, loss_function,
                      val_metric, patience, model_name, predict_neutral_class, model_id, display_fig, label_type,
                      device):

    x_train = train_list[0]  # list of lists of 3 arrays
    y_train = train_list[1]
    seg_train = train_list[2]
    x_valid = valid_list[0]  # list of lists of 3 arrays
    y_valid = valid_list[1]
    seg_valid = valid_list[2]
    num_classes = y_train[0].shape[1]
    num_train_samples = len(y_train)
    num_valid_samples = len(y_valid)
    num_test_samples = len(test_list[1])
    total_data = num_train_samples + num_valid_samples + num_test_samples

    # Create dataloaders for training and validation data
    training_data = DatasetTorch(x_train, y_train, seg_train, device)
    valid_data = DatasetTorch(x_valid, y_valid, seg_valid, device)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=my_collate_fn)
    valid_dataloader = DataLoader(valid_data, collate_fn=my_collate_fn)
    train_size = len(train_dataloader.dataset)
    valid_size = len(valid_dataloader.dataset)

    # Create an instance of the model
    num_features_text = x_train[0][0].shape[1]
    num_features_image = x_train[0][1].shape[1]
    num_features_audio = x_train[0][2].shape[1]
    model = Model_ef(model_id, num_features_text, num_features_image, num_features_audio, num_classes, num_layers,
                     num_nodes, dropout_rate, final_activ).to(device)

    # Optimizing the model parameters
    loss_fn = nn.L1Loss()  #TODO
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #TODO

    # Parameters for ModelCheckpoint and EarlyStopping callbacks
    valid_loss_best = None
    best_is_smallest = True  # TODO infer it from the type of loss
    num_epochs_without_improvement = 0

    # Parameters for history
    history_train = []
    history_valid = []

    print("\n\n============================== Training Parameters ===========================================")
    print("\n>>> Dataset")
    print("Number training datapoints: {} ({:.2f}%)".format(num_train_samples, 100 * (num_train_samples / total_data)))
    print("Number validation datapoints: {} ({:.2f}%)".format(num_valid_samples, 100 * (num_valid_samples / total_data)))
    print("Number test datapoints: {} ({:.2f}%)".format(num_test_samples, 100 * (num_test_samples / total_data)))
    print("Number of output cells:", num_classes)
    print("Predict neutral class:", predict_neutral_class)
    print("\n>>> Model training")
    print("Task:", label_type)
    print("Library: PyTorch")
    print("Batch size:", batch_size)
    print("Number epochs:", num_epochs)
    print("Patience:", patience)
    print("Learning rate:", learning_rate)
    print("Optimizer:", optimizer_name)
    print("Loss function:", loss_function)
    print("Metric to monitor on validation data:", val_metric)
    print("\n>>> Model summary (for a sequence of length 374)")
    summary(model, [(374, 300), (374, 35), (374, 74)], device='cpu', dtypes=[torch.bfloat16, torch.bfloat16, torch.bfloat16])
    # TODO does it plot the model?

    print("\n\n================================= Model Training =============================================\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch:>3d}")
        train_loss = 0.0
        valid_loss = 0.0

        # Model training on training data
        model.train()
        num_iterations = train_size // batch_size if train_size % batch_size == 0 else train_size // batch_size + 1
        with alive_bar(num_iterations, title='Model training') as bar:
            for batch, (X, y) in enumerate(train_dataloader):
                batch_loss = 0
                for X_sample, y_sample in zip(X, y):
                    X_text, X_audio, X_image = X_sample
                    if device == 'cpu':
                        X_text = X_text.bfloat16()
                        X_audio = X_audio.bfloat16()
                        X_image = X_image.bfloat16()
                    pred = model(X_text, X_audio, X_image)
                    batch_loss += loss_fn(pred, y_sample)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                # if batch % 10 == 0:
                batch_loss, current = batch_loss.item(), batch * len(X)
                # batch_loss_current = batch_loss / batch_size
                # print(f"Batch training loss: {batch_loss_current:>7f} [{current:>5d}/{train_size:>5d}]")
                train_loss += batch_loss
                bar()
        train_loss /= train_size  # TODO check the formula

        # Model evaluation on validation data
        model.eval()
        with alive_bar(valid_size, title='Model validation') as bar:
            for idx, (X, y) in enumerate(valid_dataloader):
                for X_sample, y_sample in zip(X, y):
                    X_text, X_audio, X_image = X_sample
                    if device == 'cpu':
                        X_text = X_text.bfloat16()
                        X_audio = X_audio.bfloat16()
                        X_image = X_image.bfloat16()
                    pred = model(X_text, X_audio, X_image)
                    valid_loss += loss_fn(pred, y_sample)
                    bar()
                    # if idx % 100 == 0 or idx == valid_size-1:
                    #     current_valid_loss = valid_loss / idx if idx != 0 else valid_loss
                    #     print(f"Validation loss: {current_valid_loss:>7f} [{idx:>5d}/{valid_size:>5d}]")
        valid_loss = valid_loss.item()
        valid_loss /= valid_size

        print("---------------------------------------------------------------------------------------")
        print(f"Epoch {epoch:>3d} - Training loss {train_loss:>5f}, Validation loss {valid_loss:>5f}")
        print("---------------------------------------------------------------------------------------")

        # Display and save history plot
        history_train += [train_loss]
        history_valid += [valid_loss]

        # For model checkpoint
        valid_loss_best = model_checkpoint(valid_loss, valid_loss_best, best_is_smallest, model, optimizer, epoch,
                                           train_loss, model_name, model_id)

        # For early stopping
        num_epochs_without_improvement = model_early_stopping(valid_loss, valid_loss_best, best_is_smallest,
                                                              num_epochs_without_improvement)
        # TODO Verify that it actually breaks
        if patience == num_epochs_without_improvement-1:
            # TODO create and display (or not) history + save it
            break

    # TODO Plot model history
    model_folder = os.path.join('results', model_name)
    plot_history(history_train, history_valid, model_id, val_metric,
                 history_folder="history", root_folder=model_folder, display_fig=display_fig)

    return 0
