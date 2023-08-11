from src.pytorch_utils.dataset_loader import DatasetTorch, my_collate_fn
from src.hyperparam_processing import get_model_module, get_loss_function, get_optimizer

import os
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchviz import make_dot
# import subprocess
from torch.utils.tensorboard import SummaryWriter
from alive_progress import alive_bar

from matplotlib import pyplot as plt


def inspect_model_tensorboard(writer, train_dataloader, model, model_id, model_name):
    dataiter = iter(train_dataloader)
    data, labels = next(dataiter)
    data_text, data_audio, data_image = data[0]
    data_text, data_audio, data_image = data_text.bfloat16(), data_audio.bfloat16(), data_image.bfloat16()
    writer.add_graph(model, (data_text, data_audio, data_image))
    #
    # print("hello")
    # # torchviz
    # y = model(data_text, data_audio, data_image)
    # model_path_torchviz = os.path.join('results', model_name, 'models', "model_{}".format(model_id))
    # make_dot(y, params=dict(model.named_parameters())).render(model_path_torchviz, format="dot")
    # print("hello here")
    # p = subprocess.run(["dot", "-Tpng", model_path_torchviz, ">", model_path_torchviz+".png"])
    # print("hello back")


def model_checkpoint(valid_loss, valid_loss_best, model, optimizer, epoch, train_loss,
                     model_name, model_id):

    model_folder = os.path.join('results', model_name, 'models')

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if epoch == 0 or valid_loss < valid_loss_best:

        valid_loss_best = valid_loss

        model_path = os.path.join(model_folder, "model_{}.pt".format(model_id))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, model_path)

    return valid_loss_best


def model_early_stopping(valid_loss, valid_loss_best, num_epochs_without_improvement):
    if valid_loss > valid_loss_best:
        num_epochs_without_improvement += 1
    else:
        num_epochs_without_improvement = 0
    return num_epochs_without_improvement


def plot_history(history_train, history_valid, model_id, history_folder, root_folder, display_fig):
    """
    Plot history, display and save figures

    :param history_train: List of training losses
    :param history_valid: List of validation losses
    :param model_id: Model id (int)
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
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    history_plot_path = os.path.join(full_path_folder, 'history_{}_loss.png'.format(model_id))
    plt.savefig(history_plot_path, bbox_inches='tight')
    if display_fig:
        plt.show()


def train_torch_model(train_list, valid_list, test_list, param_archi, param_training, model_name, model_id,
                      predict_neutral_class, display_fig, label_type, device):

    num_epochs = param_training['num_epochs']
    patience = param_training['patience']
    batch_size = param_training['batch_size']
    optimizer_name = param_training['optimizer']
    loss_function_str = param_training['loss_function']
    learning_rate = param_training['learning_rate']

    writer_path = os.path.join('runs', model_name, 'model_{}'.format(model_id))
    writer = SummaryWriter(writer_path)

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
    train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=True)
    valid_dataloader = DataLoader(valid_data, collate_fn=my_collate_fn)
    train_size = len(train_dataloader.dataset)
    valid_size = len(valid_dataloader.dataset)

    # Create an instance of the model
    num_features_text = x_train[0][0].shape[1]
    num_features_image = x_train[0][1].shape[1]
    num_features_audio = x_train[0][2].shape[1]
    model_class = get_model_module(model_name, is_tf=False)
    model = model_class.MModel(model_id, param_archi, num_features_text, num_features_image, num_features_audio,
                               num_classes).to(device)

    # Inspect model using Tensorboard
    inspect_model_tensorboard(writer, train_dataloader, model, model_id, model_name)

    # Optimizing the model parameters
    loss_fn = get_loss_function(loss_function_str, is_tf=False)
    optimizer = get_optimizer(optimizer_name, learning_rate, is_tf=False, model=model)
    if "when" in param_training.keys():
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=param_training["when"], factor=0.1, verbose=True)

    # Parameters for ModelCheckpoint and EarlyStopping callbacks
    valid_loss_best = None
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
    print("Loss function:", loss_function_str)
    print("\n>>> Model parameters")
    model.print_values()
    print("\n")
    print("\n>>> Model summary (for a sequence of length 374)")
    summary(model, [(374, 300), (374, 35), (374, 74)], device='cpu', dtypes=[torch.bfloat16, torch.bfloat16, torch.bfloat16])

    print("\n\n================================= Model Training =============================================\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch:>3d}")
        train_loss = 0.0
        valid_loss = 0.0

        # Model training on training data
        model.train()
        num_iterations = train_size // batch_size if train_size % batch_size == 0 else train_size // batch_size + 1
        with alive_bar(num_iterations, title='Model training') as bar:
            for batch_id, (X, y) in enumerate(train_dataloader):
                batch_loss = 0
                # X_text_batch, X_audio_batch, X_image_batch = [], [], []
                for X_sample, y_sample in zip(X, y):
                    X_text, X_audio, X_image = X_sample
                    if device == 'cpu':
                        X_text = X_text.bfloat16()
                        X_audio = X_audio.bfloat16()
                        X_image = X_image.bfloat16()
                #         X_text_batch += X_text
                #         X_audio_batch += X_audio
                #         X_image_batch += X_image
                # X_text = torch.stack(X_text_batch)
                # X_audio = torch.stack(X_audio_batch)
                # X_image = torch.stack(X_image_batch)
                # print("X_text.size():", X_text.size())
                    pred = model(X_text, X_audio, X_image)
                    batch_loss += loss_fn(pred, y_sample)

                optimizer.zero_grad()
                batch_loss.backward()
                if "clip" in param_training.keys():
                    torch.nn.utils.clip_grad_norm(model.parameters(), param_archi["clip"])
                optimizer.step()

                batch_loss, current = batch_loss.item(), batch_id * batch_size + len(X)
                batch_loss_current = batch_loss / len(X)

                if batch_id % 10 == 0 or batch_id == num_iterations-1:
                    # # Log the current batch loss in Tensorboard
                    # writer.add_scalar("Training loss (per batch)", batch_loss_current, epoch + batch_id/num_iterations)
                    writer.add_scalar("Training loss (per batch)", batch_loss_current, epoch * train_size + batch_id)
                    # Print every 10 batch losses on the terminal
                    print(f"Batch training loss: {batch_loss_current:>7f} [{current:>5d}/{train_size:>5d}]")

                train_loss += batch_loss
                bar()
        train_loss /= train_size

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

        # Log training and validation losses to Tensorboard  ##TODO Verify if the graphs are correct
        writer.add_scalar("Training loss", train_loss, epoch)
        writer.add_scalar("Validation loss", valid_loss, epoch)

        # Reduce learning rate when valid_loss stops improving
        if "when" in param_training.keys():
            scheduler.step(valid_loss)

        print("---------------------------------------------------------------------------------------")
        print(f"Epoch {epoch:>3d} - Avg training loss {train_loss:>5f}, Avg validation loss {valid_loss:>5f}")
        print("---------------------------------------------------------------------------------------")

        # Display and save history plot
        history_train += [train_loss]
        history_valid += [valid_loss]

        # For model checkpoint
        valid_loss_best = model_checkpoint(valid_loss, valid_loss_best, model, optimizer, epoch, train_loss, model_name,
                                           model_id)

        # For early stopping
        num_epochs_without_improvement = model_early_stopping(valid_loss, valid_loss_best,
                                                              num_epochs_without_improvement)
        if patience == num_epochs_without_improvement-1:
            break

    model_folder = os.path.join('results', model_name)
    plot_history(history_train, history_valid, model_id, history_folder="history", root_folder=model_folder,
                 display_fig=display_fig)

    writer.close()

    return 0
