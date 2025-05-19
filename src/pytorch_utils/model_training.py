from src.pytorch_utils.dataset_loader import DatasetTorch, collate_fn_for_variable_length
from src.hyperparam_processing import get_model_module, get_loss_function, get_val_metrics, get_optimizer
from src.print_functions import print_training_parameters, print_training_progress, print_summary_training_epoch
from src.pickle_functions import save_with_pickle
from src.model_metrics import compute_best_thres_for_monitor_metric

import os
import torch
import time
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.autograd import Variable
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


def model_checkpoint(valid_loss, valid_loss_best, valid_thres_ml_per_emo, valid_thres_ml_best,
                     model, optimizer, epoch, train_loss, model_name, ext_model_name):

    model_folder = os.path.join('results', model_name, 'models')

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if epoch == 0 or valid_loss < valid_loss_best:
        valid_thres_ml_best = valid_thres_ml_per_emo
        valid_loss_best = valid_loss

        model_path = os.path.join(model_folder, "model_{}.pt".format(ext_model_name))
        if type(optimizer) == dict:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_coa_state_dict': optimizer['coarse_task'].state_dict(),
                'optimizer_fin_state_dict': optimizer['fine_task'].state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss
            }, model_path)
        else:
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
    legend_text = ['train', 'val'] if history_train is not None else ['val']
    if type(history_valid) == dict:
        for task_j in ['coarse_task', 'fine_task']:
            if history_train is not None:
                plt.plot(history_train[task_j])
            plt.plot(history_valid[task_j])
            plt.title('Model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(legend_text)
            history_plot_path = os.path.join(full_path_folder, 'history_{}_loss_{}.png'.format(ext_name, task_j))
            plt.savefig(history_plot_path, bbox_inches='tight')
            if display_fig:
                plt.show()
            plt.clf()
    else:
        if history_train is not None:
            if type(history_train) == dict:  # In the case of functional bilevel optimization
                legend_text = ['train_fin', 'train_coa', 'val']
                plt.plot(history_train['fine_task'])
                plt.plot(history_train['coarse_task'])
            else:
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

    time_list = []

    batch_size = param_training['batch_size']
    optimizer_name = param_training['optimizer']
    num_epochs = param_training['num_epochs']
    patience = param_training['patience']
    num_training_prints = param_log['num_training_prints']
    round_decimals = param_log['round_decimals']
    display_fig = param_log['display_fig']
    one_hot = ("oh" in label_type) if type(label_type) != dict else \
        {'coarse_task': "oh" in label_type['coarse_task'], 'fine_task': "oh" in label_type['fine_task']}
    ext_model_name = "{}_fold_{}".format(model_id, fold_id) if with_cross_valid else model_id
    label_type_is_dict = (type(label_type) == dict)
    do_bilevel_opt = param_training['bilevel_opt']
    do_func_bilevel = param_training['func_bilevel']

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
    if label_type_is_dict:
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
    val_metrics_is_none = val_metrics is None
    optimizer = get_optimizer(optimizer_name, param_training, model, is_tf=False)
    if param_training["when"] != -1:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=param_training["when"], factor=0.1, verbose=True)

    # Parameters for ModelCheckpoint and EarlyStopping callbacks
    val_metrics_best, valid_thres_ml_per_emo, valid_thres_ml_best = None, None, None
    num_epochs_without_improvement = 0

    # Parameters for history
    history_train_loss = {'coarse_task': [], 'fine_task': []} if label_type_is_dict and do_bilevel_opt else []
    history_valid_loss = {'coarse_task': [], 'fine_task': []} if label_type_is_dict and do_bilevel_opt else []
    history_train_monitor = {'coarse_task': [], 'fine_task': []} if label_type_is_dict and (do_bilevel_opt and do_func_bilevel) else []
    history_valid_monitor = []

    def train_batch(x_text, x_image, x_audio, y_batch, func_bilevel_coa):

        param_model_coa, param_model_fin = optimizer['param_coarse'], optimizer['param_fine']

        def perform_param_bilevel():
            param_model_coa_names = optimizer['param_coarse_names']

            def solve_lower_pb(for_grad_approx=False):

                # This block avoid affecting the main model when performing little perturbation for grad approximation
                if for_grad_approx:
                    model_copy = model_class.MModel(model_id, param_model, param_task, num_features_text,
                                                    num_features_image, num_features_audio, num_classes).to(device)
                    model_copy.load_state_dict(model.state_dict())
                    model_copy.train()

                for i_lower in range(param_training['num_steps_bilevel_fin']):
                    #print("[i= ", i, "] lower_batch_loss:", lower_batch_loss.item())
                    optimizer['fine_task'].zero_grad()
                    pred_fine = model(x_text, x_image, x_audio)['fine_task'] # \
                        #if not for_grad_approx else model_copy(x_text, x_image, x_audio)['fine_task']
                    #print("[Fine, i:", i_lower, ", after step]\n pred_fine:", pred_fine[0], "\n y_batch:", y_batch['fine_task'][0])
                    lower_batch_loss = loss_fn['fine_task'](pred_fine, y_batch['fine_task'])
                    lower_batch_loss.backward()
                    if param_training["clip"] != -1:
                        torch.nn.utils.clip_grad_norm_(param_model_fin, param_training["clip"])
                    # Verify that it's actually training
                    #for param_name, param in model.named_parameters():
                    #    print(param_name, ", param.grad:", param.grad)
                    optimizer['fine_task'].step()
                    # if lower_batch_loss.item() < 1e-4:  # Stop if loss is small enough
                    #     break

                # Compute pred_fine at the end after lower-level model parameter update
                pred_fine = model(x_text, x_image, x_audio)['fine_task'] # \
                    # if not for_grad_approx else model_copy(x_text, x_image, x_audio)['fine_task']
                return pred_fine.detach(), lower_batch_loss

            def approximate_gradient(epsilon=1e-4):
                """
                Computes the approximate gradient of the upper-level objective with respect to the upper-level model
                parameters using finite differences.
                """
                grads_res = []
                for param_name_i, param_i in model.named_parameters():
                    if param_name_i in param_model_coa_names:  # we only update the parameters linked to the coarse task
                        original_value = param_i.data.clone()

                        # Compute upper-level loss with a small positive perturbation
                        param_i.data += epsilon
                        # Compute lower solution prediction and binarisation for multilabel classification
                        lower_solution_plus, _ = solve_lower_pb(for_grad_approx=True)
                        #lower_solution_plus_bin = (lower_solution_plus >= param_task['thres_score_pres']).float()
                        # Compute upper solution prediction
                        pred_upper_model = model(x_text, x_image, x_audio)['coarse_task']
                        #loss_plus = loss_fn['coarse_task'](pred_upper_model, lower_solution_plus_bin)
                        loss_plus = loss_fn['coarse_task'](torch.sigmoid(pred_upper_model*lower_solution_plus), y_batch['coarse_task'])

                        # Compute upper-level loss with a small negative perturbation
                        param_i.data -= 2 * epsilon
                        # Compute lower solution prediction and binarisation for multilabel classification
                        lower_solution_minus, _ = solve_lower_pb(for_grad_approx=True)
                        #lower_solution_minus_bin = (lower_solution_minus >= param_task['thres_score_pres']).float()
                        # Compute upper solution prediction
                        pred_upper_model = model(x_text, x_image, x_audio)['coarse_task']
                        #loss_minus = loss_fn['coarse_task'](pred_upper_model, lower_solution_minus_bin)
                        loss_minus = loss_fn['coarse_task'](torch.sigmoid(pred_upper_model*lower_solution_minus), y_batch['coarse_task'])

                        # Restore the original parameter value
                        param_i.data = original_value

                        # Compute the approximate gradient
                        grad_val = (loss_plus - loss_minus) / (2 * epsilon)
                        grads_res.append(grad_val)

                return grads_res

            for i_upper in range(param_training['num_steps_bilevel_coa']):  # 100 and 10 iterations for heuristics

                # The following line computes heuristically the prediction and loss for fine task
                pred_fine_best, lower_batch_loss = solve_lower_pb()
                pred_fine_best_bin = (pred_fine_best >= param_task['thres_score_pres']).float()
                pred_coarse = model(x_text, x_image, x_audio)['coarse_task']

                optimizer['coarse_task'].zero_grad()
                # # if param_training['approx_grad']:
                # #     # approximation of the gradient
                # grads = approximate_gradient()
                # param_model_coa_grads = [param_i for param_name_i, param_i in model.named_parameters()
                #                          if param_name_i in param_model_coa_names]
                # for param_i, grad in zip(param_model_coa_grads, grads):
                #     param_i.grad = grad

                # # else:
                # Compute exact gradient
                upper_batch_loss = loss_fn['coarse_task'](pred_coarse, pred_fine_best_bin)
                #upper_batch_loss = loss_fn['coarse_task'](torch.sigmoid(pred_coarse*pred_fine_best), y_batch['coarse_task'])
                upper_batch_loss.backward()
                if param_training['clip'] != -1:
                    torch.nn.utils.clip_grad_norm_(optimizer['param_coarse'], param_training["clip"])
                optimizer['coarse_task'].step()

                #print("[Coarse, i_upper:", i_upper, ", after step]\n pred_coarse:", torch.sigmoid(pred_coarse*pred_fine_best)[0], "\n y_batch:", y_batch['coarse_task'][0], "\n\n")
            batch_loss_i = {'fine_task': lower_batch_loss.item(), 'coarse_task': upper_batch_loss.item()}
            pred_batch_i = model(x_text, x_image, x_audio)

            return batch_loss_i, pred_batch_i

        def perform_optimization():
            optimizer.zero_grad()
            pred_batch = model(x_text, x_image, x_audio)
            batch_loss_i = loss_fn(pred_batch, y_batch)
            batch_loss_i.backward()
            if param_training["clip"] != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), param_training["clip"])
            optimizer.step()
            return batch_loss_i.item(), pred_batch

        def perform_functional_bilevel():
            task_type = 'coarse_task' if func_bilevel_coa else 'fine_task'
            param_model_func = param_model_coa if func_bilevel_coa else param_model_fin
            optimizer[task_type].zero_grad()
            pred_batch = model(x_text, x_image, x_audio)
            batch_loss_i = loss_fn[task_type](pred_batch[task_type], y_batch[task_type])
            batch_loss_i.backward()
            if param_training["clip"] != -1:
                torch.nn.utils.clip_grad_norm_(param_model_func, param_training["clip"])
            optimizer[task_type].step()
            return batch_loss_i.item(), pred_batch


        if do_bilevel_opt:
            if do_func_bilevel:
                return perform_functional_bilevel()
            else:
                return perform_param_bilevel()
        else:
            # classic optimization (no bilevel)
            return perform_optimization()


    def add_to_train_loss_and_monitor(train_loss_i, train_monitor_i, batch_loss_i, batch_monitor_i, func_bilevel_coa):
        # Compute training loss
        if label_type_is_dict and do_bilevel_opt:
            if func_bilevel_coa is not None:
                task_i = 'coarse_task' if func_bilevel_coa else 'fine_task'
                train_loss_i[task_i] += batch_loss_i
            else:
                for task_i in ['coarse_task', 'fine_task']:
                    train_loss_i[task_i] += batch_loss_i[task_i]
        else:
            train_loss_i += batch_loss_i

        # Compute training monitor metric
        if val_metrics is not None:
            if label_type_is_dict and do_bilevel_opt and func_bilevel_coa is not None:
                task_i = 'coarse_task' if func_bilevel_coa else 'fine_task'
                train_monitor_i[task_i] += batch_monitor_i
            else:
                train_monitor_i += batch_monitor_i

        return train_loss_i, train_monitor_i


    def add_to_valid_loss_and_monitor(valid_loss_i, valid_monitor_i, pred_i, y_i):
        pred_valid_i = {'coarse_task': [], 'fine_task': []} if label_type_is_dict else []
        true_valid_i = {'coarse_task': [], 'fine_task': []} if label_type_is_dict else []

        if label_type_is_dict:
            if do_bilevel_opt:
                for task_i in ['coarse_task', 'fine_task']:
                    valid_loss_i[task_i] += loss_fn[task_i](pred_i[task_i], y_i[task_i]).item()
            else:
                valid_loss_i += loss_fn(pred_i, y_i).item()
            for task_i in ['coarse_task', 'fine_task']:
                pred_valid_i[task_i] += pred_i[task_i].tolist()
                true_valid_i[task_i] += y_i[task_i].tolist()
        else:
            valid_loss_i += loss_fn(pred_i, y_i).item()
            pred_valid_i += pred_i.tolist()
            true_valid_i += y_i.tolist()
        if val_metrics is not None:
            valid_monitor_i += val_metrics(pred_i, y_i).item()

        return valid_loss_i, valid_monitor_i, pred_valid_i, true_valid_i


    def divide_epoch_loss_and_monitor(epoch_loss_i, epoch_monitor_i, num_iter, func_bilevel_coa):
        # Compute training loss
        if label_type_is_dict and do_bilevel_opt:
            task_list = ['coarse_task', 'fine_task'] if func_bilevel_coa is None else \
                ['coarse_task'] if func_bilevel_coa else ['fine_task']
            for task_i in task_list:
                epoch_loss_i[task_i] /= num_iter
        else:
            epoch_loss_i /= num_iter

        # Compute training monitor metric
        if val_metrics is not None:
            if label_type_is_dict and do_bilevel_opt and func_bilevel_coa is not None:
                task_i = 'coarse_task' if func_bilevel_coa else 'fine_task'
                epoch_monitor_i[task_i] /= num_iter
            else:
                epoch_monitor_i /= num_iter

        return epoch_loss_i, epoch_monitor_i


    def save_epoch_loss_and_monitor(epoch_loss_i, epoch_monitor_i, history_loss_i, history_monitor_i, func_bilevel_coa,
                                    is_train):
        # Compute training loss
        if label_type_is_dict:
            if func_bilevel_coa is not None:
                task_i = 'coarse_task' if func_bilevel_coa else 'fine_task'
                history_loss_i[task_i] += [epoch_loss_i[task_i]]
            else:
                for task_i in ['coarse_task', 'fine_task']:
                    history_loss_i[task_i] += [epoch_loss_i[task_i]]
        else:
            history_loss_i += [epoch_loss_i]

        # Compute training monitor metric
        if val_metrics_is_none and is_train:
            history_monitor_i = None
        else:
            if func_bilevel_coa is not None:
                task_i = 'coarse_task' if func_bilevel_coa else 'fine_task'
                history_monitor_i[task_i] += [epoch_monitor_i[task_i]]
            else:
                history_monitor_i += [epoch_monitor_i]

        return history_loss_i, history_monitor_i


    def compute_best_threshold_per_class(true_valid_i, pred_valid_i):
        compute_thres_ml = param_training['thres_ml_from_best_metric']
        valid_monitor_mean_res = None

        if label_type_is_dict:
            valid_thres_ml_per_emo_res, valid_monitor_mean_res = {}, {}
            for task_i in ['coarse_task', 'fine_task']:
                if val_metrics_is_none or (label_type[task_i] in ['present', 'dominant'] and compute_thres_ml):
                    t_valid, p_valid = true_valid_i[task_i], pred_valid_i[task_i]
                    vt, vm = compute_best_thres_for_monitor_metric(np.array(t_valid).reshape(-1, num_classes),
                                                                   np.array(p_valid).reshape(-1, num_classes),
                                                                   round_decimals)
                    valid_thres_ml_per_emo_res[task_i] = [np.round(v, 2) for v in vt.tolist()]  # List of thresholds
                    # Useful only when val_metrics is None
                    valid_monitor_mean_res[task_i] = vm  # List of validation metrics
                elif label_type[task_i] in ['present', 'dominant'] and not compute_thres_ml:
                        valid_thres_ml_per_emo_res[task_i] = [0.5 for _ in range(num_classes)]
                else:
                    valid_thres_ml_per_emo_res[task_i] = None

        else:
            if val_metrics_is_none or (label_type in ['present', 'dominant'] and compute_thres_ml):
                vt, valid_monitor_mean_res = compute_best_thres_for_monitor_metric(
                    np.array(true_valid_i).reshape(-1, num_classes), np.array(pred_valid_i).reshape(-1, num_classes),
                    round_decimals)
                valid_thres_ml_per_emo_res = [np.round(v, 2) for v in vt.tolist()]
            elif label_type in ['present', 'dominant'] and not compute_thres_ml:
                    valid_thres_ml_per_emo_res = [0.5 for _ in range(num_classes)]
            else:
                valid_thres_ml_per_emo_res = None

        return valid_thres_ml_per_emo_res, valid_monitor_mean_res


    def train_model(history_train_loss_i, history_train_monitor_i, train_loss_i, train_monitor_i, func_bilevel_coa=None):

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
                        batch_loss, pred = train_batch(x_text, x_image, x_audio, y_sample, func_bilevel_coa)
                        batch_monitor = val_metrics(pred, y_sample).item() if val_metrics is not None else None

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
                    batch_loss, pred = train_batch(x_text, x_image, x_audio, y, func_bilevel_coa)
                    batch_monitor = val_metrics(pred, y).item() if val_metrics is not None else None

                if (num_training_prints > 0) and \
                        (batch_id % (num_iterations // num_training_prints) == 0 or batch_id == num_iterations - 1):
                    # # Log the current batch loss in Tensorboard
                    # writer.add_scalar("Training loss (per batch)", batch_loss, epoch * train_size + batch_id)
                    # Print num_iterations batch losses on the terminal
                    current = batch_id * batch_size + len(X)
                    print_training_progress(loss_function_str, batch_loss, val_metrics, val_metrics_str, batch_monitor,
                                            current, train_size, param_training)

                train_loss_i, train_monitor_i = add_to_train_loss_and_monitor(train_loss_i, train_monitor_i,
                                                                              batch_loss, batch_monitor, func_bilevel_coa)
                bar()

        train_loss_i, train_monitor_i = divide_epoch_loss_and_monitor(train_loss_i, train_monitor_i, num_iterations,
                                                                      func_bilevel_coa)
        history_train_loss_i, history_train_monitor_i = save_epoch_loss_and_monitor(train_loss_i, train_monitor_i,
                                                                                    history_train_loss_i,
                                                                                    history_train_monitor_i,
                                                                                    func_bilevel_coa,
                                                                                    is_train=True)

        # # Compute best threshold per emotion for multilabel classification according to F1 score
        # if param_training['thres_ml_from_best_metric'] or val_metrics is None:
        #     train_thres_per_emo, train_monitor_per_emo = compute_best_thres_for_monitor_metric(
        #         np.array(true_train).reshape(-1, num_classes), np.array(pred_train).reshape(-1, num_classes),
        #         round_decimals)
        #
        #     if val_metrics is None:  # we use F1 score as a validation metric to monitor
        #         train_monitor_avg = np.round(np.mean(train_monitor_per_emo), round_decimals)
        #         train_monitor_all = train_monitor_per_emo + [train_monitor_avg]
        return history_train_loss_i, history_train_monitor_i, train_loss_i, train_monitor_i


    def valid_model(history_valid_loss_i, history_valid_monitor_i):

        valid_loss_i = {'coarse_task': 0.0, 'fine_task': 0.0} if label_type_is_dict and do_bilevel_opt else 0.0
        valid_monitor_i = 0.0

        with alive_bar(valid_size, title='Model validation') as bar:

            for idx, (X, y) in enumerate(valid_dataloader):

                if num_steps is None:  # Varying sequence length
                    for X_sample, y_sample in zip(X, y):
                        x_text, x_audio, x_image = X_sample
                        if device == 'cpu':
                            x_text = x_text.bfloat16()
                            x_audio = x_audio.bfloat16()
                            x_image = x_image.bfloat16()

                        pred = model(x_text, x_audio, x_image)
                        valid_loss_i, valid_monitor_i, pred_valid, true_valid = \
                            add_to_valid_loss_and_monitor(valid_loss_i, valid_monitor_i, pred, y_sample)

                else:  # Fixed sequence length
                    x_text , x_audio, x_image = X
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
                    valid_loss_i, valid_monitor_i, pred_valid, true_valid = \
                        add_to_valid_loss_and_monitor(valid_loss_i, valid_monitor_i, pred, y)

                bar()
                # if idx % 100 == 0 or idx == valid_size-1:
                #     current_valid_loss = valid_loss / idx if idx != 0 else valid_loss
                #     print(f"Validation loss: {current_valid_loss:>7f} [{idx:>5d}/{valid_size:>5d}]")

        valid_loss_i, valid_monitor_i = divide_epoch_loss_and_monitor(valid_loss_i, valid_monitor_i, valid_size,
                                                                      func_bilevel_coa=None)
        history_valid_loss_i, history_valid_monitor_i = save_epoch_loss_and_monitor(valid_loss_i, valid_monitor_i,
                                                                                    history_valid_loss_i,
                                                                                    history_valid_monitor_i,
                                                                                    func_bilevel_coa=None,
                                                                                    is_train=False)

        # If validation metric not specified, take F1 score as validation metric to monitor based on previous code block
        valid_thres_ml_per_emo_i, f1_avg_per_task = compute_best_threshold_per_class(true_valid, pred_valid)
        if val_metrics_is_none:
            coef_coa = param_training['coef_coa']
            mean_per_task = [f1_avg_per_task[task_i] for task_i in ['coarse_task', 'fine_task']]
            valid_monitor_i = np.average(mean_per_task, weights=[coef_coa, (1 - coef_coa)])

        return history_valid_loss_i, history_valid_monitor_i, valid_loss_i, valid_monitor_i, valid_thres_ml_per_emo_i


    fold_text = ": Fold {}".format(fold_id) if with_cross_valid else ""
    print("================================= Model Training{} =============================================\n"
          .format(fold_text))

    for epoch in range(num_epochs):
        time0 = time.time()
        print(f"Epoch {epoch:>3d}")

        # Model training on training data
        model.train()
        if do_bilevel_opt and do_func_bilevel:
            train_loss, train_monitor = {'coarse_task': 0.0, 'fine_task': 0.0}, {'coarse_task': 0.0, 'fine_task': 0.0}
            print("\n\nOptimizing for fine task:", label_type['fine_task'])
            history_train_loss, history_train_monitor, train_loss, train_monitor = \
                train_model(history_train_loss, history_train_monitor, train_loss, train_monitor, func_bilevel_coa=False)
            print("\n\nOptimizing for coarse task:", label_type['coarse_task'])
            history_train_loss, history_train_monitor, train_loss, train_monitor = \
                train_model(history_train_loss, history_train_monitor, train_loss, train_monitor, func_bilevel_coa=True)
        else:
            train_loss = {'coarse_task': 0.0, 'fine_task': 0.0} if label_type_is_dict and do_bilevel_opt else 0.0
            train_monitor = 0.0
            history_train_loss, history_train_monitor, train_loss, train_monitor = \
                    train_model(history_train_loss, history_train_monitor, train_loss, train_monitor)

        # Model evaluation on validation data
        model.eval()
        history_valid_loss, history_valid_monitor, valid_loss, valid_monitor, valid_thres_ml_per_emo = \
            valid_model(history_valid_loss, history_valid_monitor)

        # Reduce learning rate when valid_monitor stops improving
        if param_training["when"] != -1:
            scheduler.step(valid_monitor)

        print_summary_training_epoch(epoch, loss_function_str, train_loss, valid_loss, train_monitor, valid_monitor,
                                     val_metrics, val_metrics_str, valid_thres_ml_per_emo, label_type, param_training)

        # For model checkpoint
        val_metrics_best, valid_thres_ml_best = model_checkpoint(valid_monitor, val_metrics_best,
                                                                 valid_thres_ml_per_emo, valid_thres_ml_best,
                                                                 model, optimizer, epoch, train_loss,
                                                                 model_name, ext_model_name)

        time1 = time.time()
        duration = round(time1 - time0, 2)
        time_list.append(duration)
        print("Time for epoch completion:", duration, " seconds.\n\n")

        # For early stopping
        if patience != -1:
            num_epochs_without_improvement = model_early_stopping(valid_monitor, val_metrics_best,
                                                                  num_epochs_without_improvement, patience)
            if patience == num_epochs_without_improvement:
                break

    print("Total time training:", np.sum(time_list), " seconds.")
    model_folder = os.path.join('results', model_name)
    save_and_plot_history(history_train_loss, history_valid_loss, ext_model_name, history_folder="history_loss",
                          root_folder=model_folder, display_fig=display_fig)
    save_and_plot_history(history_train_monitor, history_valid_monitor, ext_model_name,
                          history_folder="history_monitor", root_folder=model_folder, display_fig=display_fig)

    writer.close()

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
