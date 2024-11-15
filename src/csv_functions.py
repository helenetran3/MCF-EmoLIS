import os
import csv
import pandas as pd


def create_csv_path(model_name, filename, ext_name="", with_cross_val=None, is_summary=False):
    """
    Create path to csv file.
    
    :param model_name: Name of the model
    :param filename: Root name of the csv file
    :param ext_name: Extension name for pickle
    :param with_cross_val: boolean (cross_validation or not)
    :param is_summary: boolean (path for summary or not)
    :return: Path to csv file.
    """
    # Create csv folder
    csv_folder = "" if is_summary else "csv"
    metric_folder_name = "" if with_cross_val is None else "cross_validation" if with_cross_val else "hold_out"
    model_csv_folder = os.path.join('results', model_name, csv_folder, metric_folder_name)
    if not os.path.isdir(model_csv_folder):
        os.makedirs(model_csv_folder)

    # Create filename
    csv_path = os.path.join(model_csv_folder, "{}{}.csv".format(filename, ext_name))

    return csv_path


def get_num_rows_csv(csv_path):
    """
    Get the number of rows in a csv file.

    :param csv_path: csv path (string)
    :return: number of lines
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        lines = len(list(reader))
    return lines


def get_current_model_id(model_name):
    """
    Get the current model id based on the number of rows in csv file.

    :param model_name: Name of the model currently tested
    :return: model_id
    """

    csv_path = create_csv_path(model_name, "model_ids")
    model_id = get_num_rows_csv(csv_path) if os.path.exists(csv_path) else 1

    return model_id


def write_csv(csv_path, header, data_to_save):
    """
    Write data in csv file.

    :param csv_path: Path to csv file
    :param header: List of headers for the csv file
    :param data_to_save: Lists of values to add (one row of the csv file)
    """
    if os.path.exists(csv_path):
        with open(csv_path, 'a+', newline='') as f:  # Open file in append mode
            writer = csv.writer(f)
            writer.writerow(data_to_save)

    else:
        with open(csv_path, 'w', encoding='UTF8') as f:  # Create file
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data_to_save)


def get_header_and_data_metrics(metrics, header_param, data_param, predict_neutral_class, task):
    """
    Creates two lists of strings (header and data) that will contain all the metrics for the specified task.

    :param metrics: a list of metric values
    :param header_param: a list of strings to identify the model (and the fold if applicable)
    :param data_param: a list of values to identify the model (and the fold if applicable)
    :param predict_neutral_class: Whether we predict the neutral class
    :param task: name of the task (between "regression", "score_coa", "presence", "dominant")
    :return: header, data
    """

    data_param = list(data_param)

    header, data = None, None  # For initialisation

    # List of emotions for headers
    emotions = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    if predict_neutral_class:
        emotions.append('neutral')

    if task == "regression":
        header_overall = ['mae', 'mse', 'r2']
        header_per_emo = ['{}_{}'.format(m, e) for m in header_overall for e in emotions]
        header = header_param + header_overall + header_per_emo
        data = data_param + metrics

    elif task == "score_coa" or task == "score_fin":
        metrics_emo = ['acc', 'f1', 'rec', 'prec']
        header_per_emo = ['{}_{}'.format(m, e) for m in metrics_emo for e in emotions]
        header_overall = ['acc_macro', 'f1_macro', 'rec_macro', 'prec_macro']
        header = header_param + header_overall + header_per_emo
        data = data_param + metrics

    elif task == "presence":
        metrics_emo = ['bin_acc', 'wei_acc', 'f1', 'rec', 'prec', 'roc_auc']
        header_per_emo = ['{}_{}'.format(m, e) for m in metrics_emo for e in emotions]
        header_overall = ['multilabel_acc', 'bin_acc_macro', 'wei_acc_macro', 'f1_macro', 'rec_macro', 'prec_macro',
                          'roc_auc_macro']
        header = header_param + header_overall + header_per_emo
        data = [data_param + m for m in metrics]  # list for different thresholds

    elif task == "dominant":
        metrics_emo = ['bin_acc', 'wei_acc', 'f1', 'rec', 'prec', 'roc_auc']
        header_per_emo = ['{}_{}'.format(m, e) for m in metrics_emo for e in emotions]
        header_overall = ['multilabel_acc', 'bin_acc_macro', 'wei_acc_macro', 'f1_macro', 'rec_macro', 'prec_macro',
                          'roc_auc_macro']
        header = header_param + header_overall + header_per_emo
        data = data_param + metrics

    else:
        raise ValueError("{} is not a possible value for 'task' parameter in get_header_and_data function."
                         .format(task))

    return header, data


def save_results_in_csv_file(model_name, model_id, loss_function, val_metrics, model_eval_output,
                             metrics_regression, metrics_scores_all, metrics_score_coa, metrics_presence,
                             metrics_dominant, with_cross_val, predict_neutral_class, thres_score_pres):
    """
    Save results of a single model to a csv file.

    :param model_name: Name of the model currently tested
    :param model_id: Model id (int)
    :param loss_function: Loss function
    :param val_metrics: Validation metric used for monitoring
    :param model_eval_output: List of [loss function, validation metric] obtained by the model
    :param metrics_regression: List of metrics for regression
    :param metrics_scores_all: List of metrics for classifying the fine-grained presence scores of each emotion
    :param metrics_score_coa: List of metrics for classifying the coarse presence scores of each emotion
    :param metrics_presence: T lists of lists of metrics for detecting the presence/absence of an emotion
    (T is the number of thresholds set)
    :param metrics_dominant: List of metrics for detecting the dominant emotion(s)
    :param with_cross_val: boolean (cross_validation or not)
    :param predict_neutral_class: Whether we predict the neutral class
    :param thres_score_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :return: One-line results added to the csv file.
    """

    # TODO metrics dict

    ext_name = "_n" if predict_neutral_class else ""

    # Create filenames
    csv_path_regression = create_csv_path(model_name, filename="regression", ext_name=ext_name,
                                          with_cross_val=with_cross_val)
    csv_path_score_all = create_csv_path(model_name, filename="classif_score_all", ext_name=ext_name,
                                         with_cross_val=with_cross_val)
    csv_path_score_coa = create_csv_path(model_name, filename="classif_score_coa", ext_name=ext_name,
                                         with_cross_val=with_cross_val)
    csv_path_presence = [create_csv_path(model_name, filename="classif_pres_t_{}".format(thres), ext_name=ext_name,
                                         with_cross_val=with_cross_val) for thres in thres_score_pres]
    csv_path_dominant = create_csv_path(model_name, filename="classif_dom", ext_name=ext_name,
                                        with_cross_val=with_cross_val)
    
    def write_csv_one_model(eval_out, met_reg, met_fin, met_coa, met_pres, met_dom, task_mt=None, fold_id=None):
        
        # Create model parameter header and data for each csv file
        header_id = ['model_id'] if not with_cross_val else ['model_id', 'fold_id']
        header_id = header_id + [loss_function, val_metrics]
        model_id_str = str(model_id) + '_' + task_mt if task_mt is not None else str(model_id)
        data_id = [model_id] if not with_cross_val else [model_id_str, fold_id]
        data_id = data_id + eval_out

        # Regression metrics
        if met_reg is not None:
            header_regression, data_regression = get_header_and_data_metrics(met_reg, header_id, data_id,
                                                                             predict_neutral_class, task="regression")
            write_csv(csv_path_regression, header_regression, data_regression)

        # Coarse presence score metrics
        if met_fin is not None:
            header_score_fin, data_score_fin = get_header_and_data_metrics(met_fin, header_id, data_id,
                                                                           predict_neutral_class, task="score_fin")
            write_csv(csv_path_score_all, header_score_fin, data_score_fin)

        # Coarse presence score metrics
        if met_coa is not None:
            header_score_coa, data_score_coa = get_header_and_data_metrics(met_coa, header_id, data_id,
                                                                           predict_neutral_class, task="score_coa")
            write_csv(csv_path_score_coa, header_score_coa, data_score_coa)

        # Present emotion metrics
        if met_pres is not None:
            header_presence, data_presence = get_header_and_data_metrics(met_pres, header_id, data_id,
                                                                         predict_neutral_class, task="presence")
            for path, data in zip(csv_path_presence, data_presence):
                write_csv(path, header_presence, data)

        # Dominant emotion metrics
        if met_dom is not None:
            header_dominant, data_dominant = get_header_and_data_metrics(met_dom, header_id, data_id,
                                                                         predict_neutral_class, task="dominant")
            write_csv(csv_path_dominant, header_dominant, data_dominant)

        # print(len(header_regression), len(data_regression))
        # print(len(header_score_coa), len(data_score_coa))
        # print(len(header_dominant), len(data_dominant))
        # for i in range(len(metrics_presence)):
        #     print(len(header_presence), len(data_presence[i]))

    if with_cross_val:
        for fold_i, (l_val_i, m_reg_i, m_all_i, m_coa_i, m_pres_i, m_dom_i) in enumerate(zip(model_eval_output,
                                                                                             metrics_regression,
                                                                                             metrics_scores_all,
                                                                                             metrics_score_coa,
                                                                                             metrics_presence,
                                                                                             metrics_dominant)):
            if type(metrics_regression) == dict:
                for task_i in ['coarse_task', 'fine_task']:
                    write_csv_one_model(l_val_i[task_i], m_reg_i[task_i], m_all_i[task_i], m_coa_i[task_i],
                                        m_pres_i[task_i], m_dom_i[task_i], task_mt=task_i, fold_id=fold_i)
            else:
                write_csv_one_model(l_val_i, m_reg_i, m_all_i, m_coa_i, m_pres_i, m_dom_i, fold_id=fold_i)
            
    else:
        if type(metrics_regression) == dict:
            for task_i in ['coarse_task', 'fine_task']:
                write_csv_one_model(model_eval_output[task_i], metrics_regression[task_i],
                                    metrics_scores_all[task_i], metrics_score_coa[task_i],
                                    metrics_presence[task_i], metrics_dominant[task_i], task_mt=task_i)
        else:
            write_csv_one_model(model_eval_output, metrics_regression, metrics_scores_all, metrics_score_coa,
                                metrics_presence, metrics_dominant)


def save_model_param_in_csv_file(model_id, param_task, param_model, param_training, model_name, loss_function,
                                 val_metrics, with_cross_val):
    """
    Create a csv file which lists the model ids and model architecture/training parameters.

    :param model_id: Model id (int)
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param param_model: dictionary of 'architecture' hyperparameters of the model chosen (cf. param_model.json)
    :param param_training: dictionary of 'training' hyperparameters (cf. param_training_task.json)
    :param model_name: Name of the model
    :param loss_function: Name of the loss function (str)
    :param val_metrics: Name of the validation metric used for monitoring (str)
    :param with_cross_val: Whether it is cross_validation or not (boolean)
    """

    predict_neutral_class = param_task['predict_neutral_class']
    set_length = param_task['set_length']
    label_type = param_task['label_type']
    thres_score_pres = param_task['thres_score_pres']
    csv_path = create_csv_path(model_name, "model_ids")
    num_classes = 7 if predict_neutral_class else 6

    # Model architecture hyperparameters
    data_archi = list(param_model.values())
    header_archi = list(param_model.keys())
    data = [model_id, model_name] + data_archi + [num_classes]
    header = ['model_id', 'model_name'] + header_archi + ['num_classes']

    # Model training hyperparameters
    data_training = list(param_training.values())
    header_training = list(param_training.keys())
    assert len(header_training) == len(data_training)
    for i in range(len(data_training)):
        if header_training[i] == 'loss_function' and type(data_training[i]) == list:
            data_training[i] = loss_function
        if header_training[i] == 'val_metrics' and type(data_training[i]) == list:
            data_training[i] = val_metrics
    data = data + data_training + [predict_neutral_class, label_type, thres_score_pres, set_length, with_cross_val]
    header = header + header_training + ['neutral_class', 'label_type', 'thres_score_pres', 'set_length', 'cross_val']

    write_csv(csv_path, header, data)


def csv_summary(with_cross_val, param_log):
    """
    Creates a file for each task which summarizes the results of all models.
     Can be found in "summary" folder in "results" folder.

    :param with_cross_val: boolean (cross_validation or not)
    :param param_log: dictionary of 'logistics' hyperparameters (cf. param_logistics.json)
    """

    if os.path.isdir("results/"):

        # This section gets the list of all filenames (regression, classif_score_coa, etc)
        metric_folder_name = "cross_validation" if with_cross_val else "hold_out"
        list_model_names = [model_name for model_name in os.listdir("results/") if model_name != 'summary' and 
                            os.path.exists(os.path.join("results", model_name, 'csv', metric_folder_name))]
        all_csv_files = []
        for model in list_model_names:
            csv_name_current_model = [csv_name for csv_name in
                                      os.listdir(os.path.join("results/", model, "csv", metric_folder_name))
                                      if csv_name != 'model_ids.csv']
            for cn in csv_name_current_model:
                filename_without_extension = os.path.splitext(cn)[0]
                if filename_without_extension not in all_csv_files:
                    all_csv_files += [filename_without_extension]

        # This section creates and fills model data in one single file (in summary/ folder)
        for csv_file in all_csv_files:   # Loop on each task
            df = pd.DataFrame()
            csv_path_summary = create_csv_path("summary", csv_file, with_cross_val=with_cross_val, is_summary=True)
            if os.path.exists(csv_path_summary):
                os.remove(csv_path_summary)  # for file reinitialisation
            for model in list_model_names:
                csv_path_model_metrics = create_csv_path(model, csv_file, with_cross_val=with_cross_val)
                csv_path_model_id = create_csv_path(model, "model_ids", with_cross_val=None)
                if os.path.exists(csv_path_model_metrics):
                    data_metrics = pd.read_csv(csv_path_model_metrics)
                    data_param = pd.read_csv(csv_path_model_id)
                    if with_cross_val:
                        data_metrics = data_metrics.drop(columns='fold_id')
                        data_metrics = data_metrics.groupby('model_id').agg(['mean', 'std']).reset_index()
                        round_decimals = param_log['round_decimals']
                        data_metrics = data_metrics.round(round_decimals)
                    model_column = [model for _ in range(len(data_metrics.index))]
                    task_column = data_param['label_type']
                    data_metrics.insert(0, 'model', model_column, allow_duplicates=True)
                    data_metrics.insert(2, 'task', task_column, allow_duplicates=True)
                    df = pd.concat([df, data_metrics], axis=0)
            df.to_csv(csv_path_summary, index=False)
    else:
        print("No results to summarize.")
