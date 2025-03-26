from src.pytorch_utils.dataset_loader import DatasetTorch, collate_fn_for_variable_length
from src.hyperparam_processing import get_model_module, get_loss_function, get_val_metrics, get_optimizer
from src.print_functions import print_training_parameters, print_training_progress, print_summary_training_epoch
from src.pickle_functions import save_with_pickle
from src.model_metrics import compute_best_thres_for_monitor_metric

import os
import re
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
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


def model_checkpoint(valid_loss, valid_loss_best, valid_thres_ml_per_emo, valid_thres_ml_best,
                     model, optimizer, epoch, train_loss, model_name, ext_model_name):

    model_folder = os.path.join('results', model_name, 'models')

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if epoch == 0 or valid_loss < valid_loss_best:
        valid_thres_ml_best = valid_thres_ml_per_emo
        valid_loss_best = valid_loss

        model_path = os.path.join(model_folder, "model_{}.pt".format(ext_model_name))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, model_path)

    return valid_loss_best, valid_thres_ml_best


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

    :param history_train: List of training losses, or list of list of num_classes+1 metrics, '+1' being the avg (can be None)
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
    emotions = ['happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear', 'avg']
    splits = ['_train', '_val'] if history_train is not None else ['_val']
    legend_text = [emo + sp for emo in emotions for sp in splits]
    if type(history_valid) == dict or type(history_valid[0]) == list:
        if type(history_valid) == dict:
            for task_j in ['coarse_task', 'fine_task']:
                for cl_i in range(len(history_valid[task_j][0])):
                    if history_train is not None:
                        history_train_i = [ht[cl_i] for ht in history_train[task_j]]
                        plt.plot(history_train_i)
                    history_valid_i = [hv[cl_i] for hv in history_valid[task_j]]
                    plt.plot(history_valid_i)
                    plt.title('Model loss')
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(legend_text)
                    history_plot_path = os.path.join(full_path_folder,
                                                     'history_{}_loss_{}.png'.format(ext_name, task_j))
                    plt.savefig(history_plot_path, bbox_inches='tight')
                    if display_fig:
                        plt.show()
    else:
        if type(history_valid[0]) == list:
            for cl_i in range(len(history_valid[0])):
                if history_train is not None:
                    history_train_i = [ht[cl_i] for ht in history_train]
                    plt.plot(history_train_i)
                history_valid_i = [hv[cl_i] for hv in history_valid]
                plt.plot(history_valid_i)
        else:
            legend_text = ['train', 'val'] if history_train is not None else ['val']
            if history_train is not None:
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


def run_one_training(train_list, valid_list, param_model, param_training, param_task, param_log,
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
    one_hot = ("oh" in label_type) if type(label_type) != dict else \
        {'coarse_task': "oh" in label_type['coarse_task'], 'fine_task': "oh" in label_type['fine_task']}
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
    if type(label_type) == dict:
        num_classes = y_train['fine_task'][0].shape[1] if not one_hot['fine_task'] else y_train['fine_task'][0].shape[0]
    else:
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
    model = model_class.MModel(model_id, param_model, param_task, num_features_text, num_features_image,
                               num_features_audio, num_classes).to(device)

    # # Inspect model using Tensorboard
    # inspect_model_tensorboard(writer, train_dataloader, model, model_id, model_name, num_steps)

    # Optimizing the model parameters
    loss_fn = get_loss_function(loss_function_str, label_type, param_training, is_tf=False)
    val_metrics = get_val_metrics(val_metrics_str, label_type, param_training, is_tf=False)
    optimizer = get_optimizer(optimizer_name, is_tf=False, param_training=param_training)
    if param_training["when"] != -1:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=param_training["when"], factor=0.1, verbose=True)

    # Parameters for ModelCheckpoint and EarlyStopping callbacks
    val_metrics_best, valid_thres_ml_per_emo, valid_thres_ml_best = None, None, None
    num_epochs_without_improvement = 0

    # Parameters for history
    history_train = []
    history_valid = []
    history_train_monitor = []
    history_valid_monitor = []

    def backprop_batch_loss(pred_batch, y_batch):

        def perform_bilevel_opt():

            num_steps_bilevel = param_training['num_steps_bilevel']

            param_coarse = [param for name, param in model.named_parameters() if '.0.' in name]
            # the following two lines remove the last layers of 'coarse' task architecture
            # (these last layers start with 'out_' and contains '.0.')
            regex_coa_final = re.compile(r'out_.*\.0\..*')
            param_coa_for_fine = [param for name, param in model.named_parameters()
                                  if ('.0.' in name and not regex_coa_final.search(name))]
            param_fine = param_coa_for_fine + [param for name, param in model.named_parameters()
                                               if '.1.' in name]
            upper_opt = optimizer['coarse_task'](param_coarse, lr=learning_rate)
            lower_opt = optimizer['fine_task'](param_fine, lr=learning_rate)

            import time
            time0 = time.time()
            #for i in range(10):  # 100 and 10 iterations for heuristics
            for i in range(num_steps_bilevel):
                if i != 0:
                    pred['fine_task'] = Variable(model(x_text, x_image, x_audio)['fine_task'], requires_grad=True)
                lower_opt.zero_grad()
                lower_batch_loss = loss_fn['fine_task'](pred_batch['fine_task'], y_batch['fine_task'])
                print("[i= ", i, "] lower_batch_loss:", lower_batch_loss.item())
                if param_training["clip"] != -1:
                    torch.nn.utils.clip_grad_norm_(param_fine, param_training["clip"])
                lower_batch_loss.backward()
                lower_opt.step()
                if lower_batch_loss.item() < 1e-4:  # Stop if loss is small enough
                    break
                #print("[Fine, i:", i, ", j:", j, ", after step]\n pred_batch:", pred_batch['fine_task'][0], "\n y_batch:", y_batch['fine_task'][0])

            pred_batch['fine_task'] = pred_batch['fine_task'].detach()
            pred_batch['coarse_task'] = Variable(model(x_text, x_image, x_audio)['coarse_task'], requires_grad=True)
            upper_opt.zero_grad()
            upper_batch_loss = loss_fn['coarse_task'](torch.sigmoid(pred_batch['coarse_task']*pred_batch['fine_task']), y_batch['coarse_task'])
            if param_training["clip"] != -1:
                torch.nn.utils.clip_grad_norm_(param_coarse, param_training["clip"])
            upper_batch_loss.backward()
            upper_opt.step()
            #print("[Coarse, i:", i, ", j:", j, ", after step]\n pred_batch:", pred_batch['coarse_task'][0], "\n y_batch:", y_batch['coarse_task'][0], "\n\n")
            batch_loss = {'fine_task': lower_batch_loss.item(), 'coarse_task': upper_batch_loss.item()}
            time1 = time.time()
            print("Time for bilevel opt:", time1 - time0, " seconds.")

            return batch_loss

        if param_training['bilevel_opt']:
            batch_loss = perform_bilevel_opt()
        else:
            optimizer.zero_grad()
            batch_loss = loss_fn(pred_batch, y_batch)
            if param_training["clip"] != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), param_training["clip"])
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.item()

        return batch_loss


    def compute_batch_monitor(pred_batch, y_batch):
        if type(label_type) == dict:
            batch_monitor = {}
            for task_i in ['coarse_task', 'fine_task']:
                batch_monitor[task_i] = val_metrics[task_i](pred_batch[task_i], y_batch[task_i]).item()
        else:
            batch_monitor = val_metrics(pred_batch, y_batch).item()
        return batch_monitor


    def add_to_train_loss_and_monitor(train_loss_i, train_monitor_i, batch_loss_i, batch_monitor_i):
        if type(label_type) == dict:
            for task_i in ['coarse_task', 'fine_task']:
                train_loss_i[task_i] += batch_loss_i[task_i]
                if val_metrics is not None:
                    train_monitor_i[task_i] += batch_monitor_i[task_i]
        else:
            train_loss_i += batch_loss_i
            if val_metrics is not None:
                train_monitor_i += batch_monitor_i

        return train_loss_i, train_monitor_i


    def divide_train_loss_and_monitor(train_loss_i, train_monitor_i, num_iter):
        if type(label_type) == dict:
            for task_i in ['coarse_task', 'fine_task']:
                train_loss_i[task_i] /= num_iter
                if val_metrics is not None:
                    train_monitor_i[task_i] /= num_iter
        else:
            train_loss_i /= num_iter
            if val_metrics is not None:
                train_monitor_i /= num_iter

        return train_loss_i, train_monitor_i


    fold_text = ": Fold {}".format(fold_id) if with_cross_valid else ""
    print("================================= Model Training{} =============================================\n"
          .format(fold_text))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch:>3d}")
        train_loss = {'coarse_task': 0.0, 'fine_task': 0.0} if type(label_type) == dict else 0.0
        valid_loss = {'coarse_task': 0.0, 'fine_task': 0.0} if type(label_type) == dict else 0.0
        train_monitor = {'coarse_task': 0.0, 'fine_task': 0.0} if type(label_type) == dict else 0.0
        valid_monitor = {'coarse_task': 0.0, 'fine_task': 0.0} if type(label_type) == dict else 0.0

        # Model training on training data
        model.train()
        num_iterations = train_size // batch_size if train_size % batch_size == 0 else train_size // batch_size + 1
        with alive_bar(num_iterations, title='Model training') as bar:
            for batch_id, (X, y) in enumerate(train_dataloader):

                if num_steps is None:  # Varying sequence length
                    for X_sample, y_sample in zip(X, y):
                        x_text, x_image, x_audio = X_sample
                        if device == 'cpu':
                            x_text = x_text.bfloat16()
                            x_image = x_image.bfloat16()
                            x_audio = x_audio.bfloat16()

                        # Prediction and compute training metrics
                        pred = model(x_text, x_image, x_audio)
                        batch_loss = backprop_batch_loss(pred, y_sample)
                        if val_metrics is not None:
                            batch_monitor = compute_batch_monitor(pred, y_sample)

                else:  # Fixed sequence length
                    x_text, x_image, x_audio = X
                    if device == 'cpu':
                        x_text = x_text.float()
                        x_image = x_image.float()
                        x_audio = x_audio.float()
                    if type(y) == dict:
                        for task_i in ['coarse_task', 'fine_task']:
                            y[task_i] = y[task_i].float()
                    else:
                        y = y.float()

                    # Prediction and compute training metrics
                    pred = model(x_text, x_image, x_audio)
                    batch_loss = backprop_batch_loss(pred, y)
                    if val_metrics is not None:
                        batch_monitor = compute_batch_monitor(pred, y)

                if (num_training_prints > 0) and \
                        (batch_id % (num_iterations // num_training_prints) == 0 or batch_id == num_iterations - 1):
                    # # Log the current batch loss in Tensorboard
                    # writer.add_scalar("Training loss (per batch)", batch_loss, epoch * train_size + batch_id)
                    # Print num_iterations batch losses on the terminal
                    current = batch_id * batch_size + len(X)
                    print_training_progress(loss_function_str, batch_loss, val_metrics, val_metrics_str, batch_monitor,
                                            current, train_size)

                train_loss, train_monitor = add_to_train_loss_and_monitor(train_loss, train_monitor,
                                                                          batch_loss, batch_monitor)
                bar()

        train_loss, train_monitor = divide_train_loss_and_monitor(train_loss, train_monitor, num_iterations)

        # # Compute best threshold per emotion for multilabel classification according to F1 score
        # if param_training['thres_ml_from_best_metric'] or val_metrics is None:
        #     train_thres_per_emo, train_monitor_per_emo = compute_best_thres_for_monitor_metric(
        #         np.array(true_train).reshape(-1, num_classes), np.array(pred_train).reshape(-1, num_classes),
        #         round_decimals)
        #
        #     if val_metrics is None:  # we use F1 score as a validation metric to monitor
        #         train_monitor_avg = np.round(np.mean(train_monitor_per_emo), round_decimals)
        #         train_monitor_all = train_monitor_per_emo + [train_monitor_avg]

        # Model evaluation on validation data
        model.eval()
        with alive_bar(valid_size, title='Model validation') as bar:
            pred_valid = {'coarse_task': [], 'fine_task': []} if type(label_type) == dict else []
            true_valid = {'coarse_task': [], 'fine_task': []} if type(label_type) == dict else []
            for idx, (X, y) in enumerate(valid_dataloader):

                if num_steps is None:  # Varying sequence length
                    for X_sample, y_sample in zip(X, y):
                        x_text, x_audio, x_image = X_sample
                        if device == 'cpu':
                            x_text = x_text.bfloat16()
                            x_audio = x_audio.bfloat16()
                            x_image = x_image.bfloat16()

                        pred = model(x_text, x_audio, x_image)
                        if type(label_type) == dict:
                            for task_i in ['coarse_task', 'fine_task']:
                                valid_loss[task_i] += loss_fn[task_i](pred[task_i], y_sample[task_i]).item()
                                if val_metrics is not None:
                                    valid_monitor[task_i] += val_metrics[task_i](pred[task_i], y_sample[task_i]).item()
                                pred_valid[task_i] += pred[task_i].tolist()
                                true_valid[task_i] += y_sample[task_i].tolist()
                        else:
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
                    if type(y) == dict:
                        for task_i in ['coarse_task', 'fine_task']:
                            y[task_i] = y[task_i].float()
                    else:
                        y = y.float()

                    pred = model(x_text, x_audio, x_image)
                    if type(label_type) == dict:
                        for task_i in ['coarse_task', 'fine_task']:
                            valid_loss[task_i] += loss_fn[task_i](pred[task_i], y[task_i]).item()
                            if val_metrics is not None:
                                valid_monitor[task_i] += val_metrics[task_i](pred[task_i], y[task_i]).item()
                            pred_valid[task_i] += pred[task_i].tolist()
                            true_valid[task_i] += y[task_i].tolist()
                    else:
                        valid_loss += loss_fn(pred, y).item()
                        if val_metrics is not None:
                            valid_monitor += val_metrics(pred, y).item()
                        pred_valid += pred.tolist()
                        true_valid += y.tolist()

                bar()
                # if idx % 100 == 0 or idx == valid_size-1:
                #     current_valid_loss = valid_loss / idx if idx != 0 else valid_loss
                #     print(f"Validation loss: {current_valid_loss:>7f} [{idx:>5d}/{valid_size:>5d}]")

        if type(label_type) == dict:
            for task_i in ['coarse_task', 'fine_task']:
                valid_loss[task_i] /= valid_size
                if val_metrics is not None:
                    valid_monitor[task_i] /= valid_size
        else:
            valid_loss /= valid_size
            if val_metrics is not None:
                valid_monitor /= valid_size

        # Compute best threshold per emotion for multilabel classification according to F1 score
        if type(label_type) == str:
            if label_type in ['present', 'dominant'] and (param_training['thres_ml_from_best_metric'] or
                                                          val_metrics is None):
                valid_thres_ml_per_emo, valid_monitor_per_emo = compute_best_thres_for_monitor_metric(
                    np.array(true_valid).reshape(-1, num_classes), np.array(pred_valid).reshape(-1, num_classes),
                    round_decimals)

        elif type(label_type) == dict:
            valid_thres_ml_per_emo, valid_monitor_per_emo_i = {}, {}
            for task_i in ['coarse_task', 'fine_task']:
                if val_metrics is None or (param_training['thres_ml_from_best_metric'] and
                                           label_type[task_i] in ['present', 'dominant']):
                    t_valid, p_valid = true_valid[task_i], pred_valid[task_i]
                    vt, vm = compute_best_thres_for_monitor_metric(np.array(t_valid).reshape(-1, num_classes),
                                                                   np.array(p_valid).reshape(-1, num_classes),
                                                                   round_decimals)
                    valid_thres_ml_per_emo[task_i] = [np.round(v, 2) for v in vt.tolist()]  # List of thresholds
                    # Used only when val_metrics is None
                    valid_monitor_per_emo_i[task_i] = vm  # List of validation metrics

        # If validation metric not specified, take F1 score as validation metric to monitor based on previous code block
        if val_metrics is None:
            if type(label_type) == dict:
                mean_per_task = [np.mean(valid_monitor_per_emo_i[task_i]) for task_i in ['coarse_task', 'fine_task']]
                valid_monitor_avg = np.round(np.mean(mean_per_task), round_decimals)
                # The following three lines is to add mean to the list of validation metrics values
                valid_monitor_all = valid_monitor_per_emo_i.copy()
                valid_monitor_all['coarse_task'] += [mean_per_task[0]]
                valid_monitor_all['fine_task'] += [mean_per_task[1]]
            else:
                valid_monitor_avg = np.round(np.mean(valid_monitor_per_emo), round_decimals)
                valid_monitor_all = valid_monitor_per_emo + [valid_monitor_avg]

            # We monitor the average for the rest of the code
            valid_monitor = valid_monitor_avg

        # # For Tensorboard
        # emotions = ['happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear', 'avg']
        # for cl_i in range(len(emotions)):
        #     writer.add_scalar("Training to monitor - {}".format(emotions[cl_i]), train_monitor_all[cl_i], epoch)
        #     writer.add_scalar("Validation to monitor - {}".format(emotions[cl_i]), valid_monitor_all[cl_i], epoch)

        # # Log training and validation losses to Tensorboard  ##TODO Verify if the graphs are correct
        # writer.add_scalar("Training loss", train_loss, epoch)
        # writer.add_scalar("Validation loss", valid_loss, epoch)
        # if val_metrics is not None:
        #     writer.add_scalar("Training to monitor", train_monitor, epoch)
        #     writer.add_scalar("Validation to monitor", valid_monitor, epoch)

        # Reduce learning rate when valid_monitor stops improving
        if param_training["when"] != -1:
            scheduler.step(valid_monitor)

        print_summary_training_epoch(epoch, loss_function_str, train_loss, valid_loss, val_metrics, val_metrics_str,
                                     valid_thres_ml_per_emo, train_monitor, valid_monitor, valid_monitor_avg,
                                     valid_monitor_per_emo, valid_monitor_per_emo_i, label_type)

        # Display and save history plot
        history_train += [train_loss]
        history_valid += [valid_loss]
        if val_metrics is None:
            # history_train_monitor += [train_monitor_all]
            history_train_monitor = None
            history_valid_monitor += [valid_monitor_all]
        else:
            history_train_monitor += [train_monitor]
            history_valid_monitor += [valid_monitor]

        # For model checkpoint
        val_metrics_best, valid_thres_ml_best = model_checkpoint(valid_monitor, val_metrics_best,
                                                                 valid_thres_ml_per_emo, valid_thres_ml_best,
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

    thres_list = [0.5 for _ in range(num_classes)]
    if type(label_type) == str:
        if label_type in ['present', 'dominant'] and not param_training['thres_ml_from_best_metric']:
            valid_thres_ml_best = thres_list
    elif type(label_type) == dict:
        for task_i, label_i in zip(label_type.keys(), label_type.values()):
            if label_i in ['present', 'dominant'] and not param_training['thres_ml_from_best_metric']:
                valid_thres_ml_best[task_i] = thres_list
            elif label_i not in ['present', 'dominant']:
                valid_thres_ml_best[task_i] = None

    print("\n\n")

    return valid_thres_ml_best


def train_torch_model(train_list, valid_list, test_list, param_task, param_model, param_training, param_log, num_steps,
                      model_name, model_id, loss_function, val_metrics, label_type, device, with_cross_valid):

    print_training_parameters(train_list, valid_list, test_list, with_cross_valid,
                              param_task, param_model, param_training, num_steps, model_name, model_id,
                              loss_function, val_metrics, label_type, is_tf=False)

    if with_cross_valid:
        valid_thres_ml_best = []
        for fold_id, (train_list_i, valid_list_i) in enumerate(zip(train_list, valid_list)):
            valid_thres_ml_best += [run_one_training(train_list_i, valid_list_i, param_model, param_training,
                                                     param_task, param_log, num_steps, model_name, model_id,
                                                     loss_function, val_metrics, label_type, device, with_cross_valid,
                                                     fold_id)]

    else:
        valid_thres_ml_best = run_one_training(train_list, valid_list, param_model, param_training, param_task,
                                               param_log, num_steps, model_name, model_id, loss_function, val_metrics,
                                               label_type, device, with_cross_valid)

    return valid_thres_ml_best
