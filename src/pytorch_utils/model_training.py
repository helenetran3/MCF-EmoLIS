from src.pytorch_utils.dataset_loader import DatasetTorch, collate_fn_for_variable_length
from src.hyperparam_processing import get_model_module, get_loss_function, get_val_metrics, get_optimizer
from src.print_functions import print_training_parameters
from src.pickle_functions import save_with_pickle
from src.model_metrics import compute_best_thres_for_monitor_metric

import os
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchviz import make_dot
# import subprocess
from torch.utils.tensorboard import SummaryWriter
from alive_progress import alive_bar

import numpy as np
from matplotlib import pyplot as plt


def inspect_model_tensorboard(writer, train_dataloader, model, model_id, model_name, num_steps):
    dataiter = iter(train_dataloader)
    data, labels = next(dataiter)
    if num_steps is None:
        data_text, data_audio, data_image = data[0]
        data_text, data_audio, data_image = data_text.bfloat16(), data_audio.bfloat16(), data_image.bfloat16()
    else:
        data_text, data_audio, data_image = data
        data_text, data_audio, data_image = data_text.float(), data_audio.float(), data_image.float()

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


def model_checkpoint(valid_loss, valid_loss_best, valid_thres_per_emo, valid_thres_per_emo_best,
                     model, optimizer, epoch, train_loss, model_name, ext_model_name):

    model_folder = os.path.join('results', model_name, 'models')

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if epoch == 0 or valid_loss < valid_loss_best:
        valid_thres_per_emo_best = valid_thres_per_emo
        valid_loss_best = valid_loss

        model_path = os.path.join(model_folder, "model_{}.pt".format(ext_model_name))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, model_path)

    return valid_loss_best, valid_thres_per_emo_best


def model_early_stopping(valid_loss, valid_loss_best, num_epochs_without_improvement, patience):
    if valid_loss > valid_loss_best:
        num_epochs_without_improvement += 1
        if patience == num_epochs_without_improvement:
            print("[Early stopping] Number epochs without improvement reaches patience {}. Model training terminates."
                  .format(patience))
        else:
            print("[Early stopping] {} epoch{} without improvement".format(num_epochs_without_improvement,
                                                                           '' if num_epochs_without_improvement == 1
                                                                           else 's'))
    else:
        print("[Early stopping] Performance improved: model saved!")
        num_epochs_without_improvement = 0
    return num_epochs_without_improvement


def save_and_plot_history(history_train, history_valid, ext_name, history_folder, root_folder, display_fig):
    """
    Save and plot history, display and save figures

    :param history_train: List of training losses, or list of list of num_classes+1 metrics, '+1' being the avg
    :param history_valid: List of validation losses, or list of list of num_classes+1 metrics, '+1' being the avg
    :param ext_name: Extension name (str)
    :param history_folder: Name of the folder where the history plot will be saved
    :param root_folder: Name of the root folder where the history folder is created
    :param display_fig: Whether we display the figures
    """

    history = [history_train, history_valid]
    save_with_pickle(history, "history_{}".format(ext_name), pickle_folder="history", root_folder=root_folder)

    # Create full path folder
    full_path_folder = os.path.join(root_folder, history_folder)
    if not os.path.isdir(full_path_folder):
        os.makedirs(full_path_folder)

    # summarize history for training and validation metric used to monitor
    if type(history_train[0]) == list:
        emotions = ['happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear', 'avg']
        splits = ['_train', '_val']
        legend_text = [emo + sp for emo in emotions for sp in splits]
        for cl_i in range(len(history_train[0])):
            history_train_i = [ht[cl_i] for ht in history_train]
            history_valid_i = [hv[cl_i] for hv in history_valid]
            plt.plot(history_train_i)
            plt.plot(history_valid_i)
    else:
        legend_text = ['train', 'val']
        plt.plot(history_train)
        plt.plot(history_valid)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legend_text)
    history_plot_path = os.path.join(full_path_folder, 'history_{}_loss.png'.format(ext_name))
    plt.savefig(history_plot_path, bbox_inches='tight')
    if display_fig:
        plt.show()


def run_one_training(train_list, valid_list, param_model, param_training, param_log,
                     num_steps, model_name, model_id, loss_function_str, val_metrics_str, label_type, device,
                     with_cross_valid, fold_id=None):

    batch_size = param_training['batch_size']
    optimizer_name = param_training['optimizer']
    learning_rate = param_training['learning_rate']
    num_epochs = param_training['num_epochs']
    patience = param_training['patience']
    num_training_prints = param_log['num_training_prints']
    round_decimals = param_log['round_decimals']
    display_fig = param_log['display_fig']
    one_hot = ("oh" in label_type)
    ext_model_name = "{}_fold_{}".format(model_id, fold_id) if with_cross_valid else model_id

    # Writer for Tensorboard
    writer_path = os.path.join('runs', model_name, 'model_{}'.format(ext_model_name))
    writer = SummaryWriter(writer_path)

    x_train = train_list[0]  # list of lists of 3 arrays
    y_train = train_list[1]
    seg_train = train_list[2]
    x_valid = valid_list[0]  # list of lists of 3 arrays
    y_valid = valid_list[1]
    seg_valid = valid_list[2]
    # In the following, predict neutral class is already inferred from label preprocessing
    num_classes = y_train[0].shape[1] if not one_hot else y_train[0].shape[0]

    # Create dataloaders for training and validation data
    training_data = DatasetTorch(x_train, y_train, seg_train, num_steps, one_hot, device)
    valid_data = DatasetTorch(x_valid, y_valid, seg_valid, num_steps, one_hot, device)
    my_collate_fn = None if num_steps is not None else collate_fn_for_variable_length
    train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=True)
    valid_dataloader = DataLoader(valid_data, collate_fn=my_collate_fn)
    train_size = len(train_dataloader.dataset)
    valid_size = len(valid_dataloader.dataset)

    # Create an instance of the model
    num_features_text = x_train[0][0].shape[1]
    num_features_image = x_train[0][1].shape[1]
    num_features_audio = x_train[0][2].shape[1]
    model_class = get_model_module(model_name, is_tf=False)
    model = model_class.MModel(model_id, param_model, num_features_text, num_features_image, num_features_audio,
                               num_classes).to(device)

    # # Inspect model using Tensorboard
    # inspect_model_tensorboard(writer, train_dataloader, model, model_id, model_name, num_steps)

    # Optimizing the model parameters
    loss_fn = get_loss_function(loss_function_str, is_tf=False)
    val_metrics = get_val_metrics(val_metrics_str, is_tf=False)
    optimizer = get_optimizer(optimizer_name, learning_rate, is_tf=False, model=model)
    if param_training["when"] != -1:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=param_training["when"], factor=0.1, verbose=True)

    # Parameters for ModelCheckpoint and EarlyStopping callbacks
    val_metrics_best, valid_thres_per_emo_best = None, None
    num_epochs_without_improvement = 0

    # Parameters for history
    history_train = []
    history_valid = []
    history_train_monitor = []
    history_valid_monitor = []

    fold_text = ": Fold {}".format(fold_id) if with_cross_valid else ""
    print("================================= Model Training{} =============================================\n"
          .format(fold_text))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch:>3d}")
        train_loss = 0.0
        valid_loss = 0.0
        train_monitor = 0.0
        valid_monitor = 0.0

        # Model training on training data
        model.train()
        num_iterations = train_size // batch_size if train_size % batch_size == 0 else train_size // batch_size + 1
        with alive_bar(num_iterations, title='Model training') as bar:
            pred_train, true_train = [], []
            for batch_id, (X, y) in enumerate(train_dataloader):

                if num_steps is None:  # Varying sequence length
                    for X_sample, y_sample in zip(X, y):
                        x_text, x_image, x_audio = X_sample
                        if device == 'cpu':
                            x_text = x_text.bfloat16()
                            x_image = x_image.bfloat16()
                            x_audio = x_audio.bfloat16()
                        pred = model(x_text, x_image, x_audio)
                        batch_loss = loss_fn(pred, y_sample)
                        if val_metrics is not None:
                            batch_monitor = val_metrics(pred, y_sample).item()
                        pred_train += pred.tolist()
                        true_train += y_sample.tolist()

                else:  # Fixed sequence length
                    x_text, x_image, x_audio = X
                    if device == 'cpu':
                        x_text = x_text.float()
                        x_image = x_image.float()
                        x_audio = x_audio.float()
                    pred = model(x_text, x_image, x_audio)
                    batch_loss = loss_fn(pred, y.float())
                    if val_metrics is not None:
                        batch_monitor = val_metrics(pred, y.float()).item()
                    pred_train += pred.tolist()
                    true_train += y.tolist()

                optimizer.zero_grad()
                batch_loss.backward()

                if param_training["clip"] != -1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), param_training["clip"])
                optimizer.step()

                batch_loss = batch_loss.item()
                current = batch_id * batch_size + len(X)
                # batch_loss_current = batch_loss / len(X)
                # batch_monitor_current = batch_monitor / len(X)

                # if batch_id % 10 == 0 or batch_id == num_iterations-1:
                if (num_training_prints > 0) and \
                        (batch_id % (num_iterations // num_training_prints) == 0 or batch_id == num_iterations - 1):
                    # # Log the current batch loss in Tensorboard
                    # writer.add_scalar("Training loss (per batch)", batch_loss_current, epoch + batch_id/num_iterations)
                    writer.add_scalar("Training loss (per batch)", batch_loss, epoch * train_size + batch_id)
                    # Print every 10 batch losses on the terminal
                    print(f"[Batch] {loss_function_str}: {batch_loss:>7f}",
                          f"- {val_metrics_str}: {batch_monitor:>7f}" if val_metrics is not None else "",
                          f"[{current:>5d}/{train_size:>5d}]")

                train_loss += batch_loss
                if val_metrics is not None:
                    train_monitor += batch_monitor
                bar()

        train_loss /= num_iterations
        if val_metrics is not None:
            train_monitor /= num_iterations

        # Compute best threshold per emotion for multilabel classification according to F1 score
        train_thres_per_emo, train_monitor_per_emo = compute_best_thres_for_monitor_metric(
            np.array(true_train).reshape(-1, num_classes), np.array(pred_train).reshape(-1, num_classes),
            round_decimals)

        if val_metrics is None:  # we use F1 score as a validation metric to monitor
            train_monitor_avg = np.round(np.mean(train_monitor_per_emo), round_decimals)
            train_monitor_all = train_monitor_per_emo + [train_monitor_avg]

        # Model evaluation on validation data
        model.eval()
        with alive_bar(valid_size, title='Model validation') as bar:
            pred_valid, true_valid = [], []
            for idx, (X, y) in enumerate(valid_dataloader):

                if num_steps is None:  # Varying sequence length
                    for X_sample, y_sample in zip(X, y):
                        x_text, x_audio, x_image = X_sample
                        if device == 'cpu':
                            x_text = x_text.bfloat16()
                            x_audio = x_audio.bfloat16()
                            x_image = x_image.bfloat16()
                        pred = model(x_text, x_audio, x_image)
                        valid_loss += loss_fn(pred, y_sample).item()
                        if val_metrics is not None:
                            valid_monitor += val_metrics(pred, y_sample).item()
                        pred_valid += pred.tolist()
                        true_valid += y_sample.tolist()

                else:  # Fixed sequence length
                    x_text, x_audio, x_image = X
                    if device == 'cpu':
                        x_text = x_text.float()
                        x_audio = x_audio.float()
                        x_image = x_image.float()
                    pred = model(x_text, x_audio, x_image)
                    valid_loss += loss_fn(pred, y.float()).item()
                    if val_metrics is not None:
                        valid_monitor += val_metrics(pred, y.float()).item()
                    pred_valid += pred.tolist()
                    true_valid += y.tolist()

                bar()
                # if idx % 100 == 0 or idx == valid_size-1:
                #     current_valid_loss = valid_loss / idx if idx != 0 else valid_loss
                #     print(f"Validation loss: {current_valid_loss:>7f} [{idx:>5d}/{valid_size:>5d}]")
        valid_loss /= valid_size
        if val_metrics is not None:
            valid_monitor /= valid_size

        # Compute best threshold per emotion for multilabel classification according to F1 score
        valid_thres_per_emo, valid_monitor_per_emo = compute_best_thres_for_monitor_metric(
            np.array(true_valid).reshape(-1, num_classes), np.array(pred_valid).reshape(-1, num_classes),
            round_decimals)

        # If validation metric not specified, take F1 score as validation metric to monitor based on previous code block
        if val_metrics is None:
            valid_monitor_avg = np.round(np.mean(valid_monitor_per_emo), round_decimals)
            valid_monitor_all = valid_monitor_per_emo + [valid_monitor_avg]
            # We monitor the average for the rest of the code
            valid_monitor = valid_monitor_avg

            # For Tensorboard
            emotions = ['happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear', 'avg']
            for cl_i in range(len(emotions)):
                writer.add_scalar("Training to monitor - {}".format(emotions[cl_i]), train_monitor_all[cl_i], epoch)
                writer.add_scalar("Validation to monitor - {}".format(emotions[cl_i]), valid_monitor_all[cl_i], epoch)

        # Log training and validation losses to Tensorboard  ##TODO Verify if the graphs are correct
        writer.add_scalar("Training loss", train_loss, epoch)
        writer.add_scalar("Validation loss", valid_loss, epoch)
        if val_metrics is not None:
            writer.add_scalar("Training to monitor", train_monitor, epoch)
            writer.add_scalar("Validation to monitor", valid_monitor, epoch)

        # Reduce learning rate when valid_monitor stops improving
        if param_training["when"] != -1:
            scheduler.step(valid_monitor)

        print("-------------------------------------------------------------------------------------------------------")
        print(f"Epoch {epoch:>3d} - Avg train {loss_function_str}: {train_loss:>5f}")
        print(f"          - Avg valid {loss_function_str}: {valid_loss:>5f}")
        if val_metrics is None:
            print(f"          - Avg train F1 score: {train_monitor_avg}; per emo: {train_monitor_per_emo}")  # TODO replace F1 by val_met
            print(f"          - Avg valid F1 score: {valid_monitor_avg}; per emo: {valid_monitor_per_emo}")
        else:
            print(f"          - Avg train {val_metrics_str}: {train_monitor:>5f}")
            print(f"          - Avg valid {val_metrics_str}: {valid_monitor:>5f}")
        if param_training['thres_ml_best']:
            print(f"          - Threshold train per emo (according to F1 score): {train_thres_per_emo}")
            print(f"          - Threshold valid per emo (according to F1 score): {valid_thres_per_emo}")
        print("-------------------------------------------------------------------------------------------------------")

        # Display and save history plot
        history_train += [train_loss]
        history_valid += [valid_loss]
        if val_metrics is None:
            history_train_monitor += [train_monitor_all]
            history_valid_monitor += [valid_monitor_all]
        else:
            history_train_monitor += [train_monitor]
            history_valid_monitor += [valid_monitor]

        # For model checkpoint
        val_metrics_best, valid_thres_per_emo_best = model_checkpoint(valid_monitor, val_metrics_best,
                                                                      valid_thres_per_emo, valid_thres_per_emo_best,
                                                                      model, optimizer, epoch, train_loss,
                                                                      model_name, ext_model_name)

        # For early stopping
        if patience != -1:
            num_epochs_without_improvement = model_early_stopping(valid_monitor, val_metrics_best,
                                                                  num_epochs_without_improvement, patience)
            if patience == num_epochs_without_improvement:
                break

    model_folder = os.path.join('results', model_name)
    save_and_plot_history(history_train, history_valid, ext_model_name, history_folder="history",
                          root_folder=model_folder, display_fig=display_fig)
    save_and_plot_history(history_train_monitor, history_valid_monitor, ext_model_name,
                          history_folder="history_monitor", root_folder=model_folder, display_fig=display_fig)

    writer.close()

    if not param_training['thres_ml_best']:
        valid_thres_per_emo_best = [0.5 for _ in range(num_classes)]

    return valid_thres_per_emo_best


def train_torch_model(train_list, valid_list, test_list, param_task, param_model, param_training, param_log, num_steps,
                      model_name, model_id, loss_function, val_metrics, label_type, device, with_cross_valid):

    print_training_parameters(train_list, valid_list, test_list, with_cross_valid,
                              param_task, param_model, param_training, num_steps, model_name, model_id,
                              loss_function, val_metrics, is_tf=False)

    if with_cross_valid:
        valid_thres_per_emo_best = []
        for fold_id, (train_list_i, valid_list_i) in enumerate(zip(train_list, valid_list)):
            valid_thres_per_emo_best += [run_one_training(train_list_i, valid_list_i, param_model, param_training,
                                                          param_log, num_steps, model_name, model_id, loss_function,
                                                          val_metrics, label_type, device, with_cross_valid, fold_id)]

    else:
        valid_thres_per_emo_best = run_one_training(train_list, valid_list, param_model, param_training, param_log,
                                                    num_steps, model_name, model_id, loss_function, val_metrics,
                                                    label_type, device, with_cross_valid)

    return valid_thres_per_emo_best
