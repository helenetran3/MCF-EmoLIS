import os
import csv


def create_csv_path(model_name, filename, extension_name=""):
    """
    Create path to csv file.
    
    :param model_name: Name of the model
    :param filename: Root name of the csv file
    :param extension_name: Extension name for pickle
    :return: Path to csv file.
    """
    # Create csv folder
    model_csv_folder = os.path.join('results', model_name, 'csv')
    if not os.path.isdir(model_csv_folder):
        os.makedirs(model_csv_folder)

    # Create filename
    csv_path = os.path.join(model_csv_folder, "{}{}.csv".format(filename, extension_name))

    return csv_path


def get_current_model_id(model_name):
    """
    Get the current model id based on the number of rows in csv file.

    :param model_name: Name of the model currently tested
    :return: model_id
    """

    def get_num_rows_csv():
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            lines = len(list(reader))
        return lines

    csv_path = create_csv_path(model_name, "model_ids")

    model_id = get_num_rows_csv() if os.path.exists(csv_path) else 1

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

    elif task == "score_coa":
        metrics_emo = ['f1_macro', 'f1_weighted', 'rec_macro', 'rec_weighted', 'prec_macro', 'prec_weighted']
        header_per_emo = ['{}_{}'.format(m, e) for m in metrics_emo for e in emotions]
        header_overall = ['acc', 'f1_macro_uw_mean', 'f1_weighted_uw_mean', 'rec_macro_uw_mean', 'rec_weighted_uw_mean',
                          'prec_macro_uw_mean', 'prec_weighted_uw_mean']
        header = header_param + header_overall + header_per_emo
        data = data_param + metrics

    elif task == "presence":
        metrics_emo = ['acc', 'f1', 'rec', 'prec']
        header_per_emo = ['{}_{}'.format(m, e) for m in metrics_emo for e in emotions]
        header_overall = ['multilabel_acc', 'f1_macro', 'f1_weighted', 'rec_macro', 'rec_weighted',
                          'prec_macro', 'prec_weighted']
        header = header_param + header_overall + header_per_emo
        data = [data_param + m for m in metrics]  # list for different thresholds

    elif task == "dominant":
        metrics_emo = ['acc', 'f1', 'rec', 'prec', 'roc_auc']
        header_per_emo = ['{}_{}'.format(m, e) for m in metrics_emo for e in emotions]
        header_overall = ['multilabel_acc', 'f1_macro', 'f1_weighted', 'rec_macro', 'rec_weighted',
                          'prec_macro', 'prec_weighted', 'roc_auc_macro', 'roc_auc_weighted']
        header = header_param + header_overall + header_per_emo
        data = data_param + metrics

    else:
        print("{} is not a possible value for 'task' parameter in get_header_and_data function."
              .format(task))

    return header, data


def save_results_in_csv_file(model_name, model_id,
                             loss_function, loss_function_val, metrics_regression, metrics_score_coa,
                             metrics_presence, metrics_dominant, predict_neutral_class, threshold_emo_pres,
                             extension_name):
    """
    Save results of a single model to a csv file.

    :param model_name: Name of the model currently tested
    :param model_id: Model id (int)
    :param loss_function: Loss function
    :param loss_function_val: Loss function obtained by the model
    :param metrics_regression: List of metrics for regression (w.r.t the presence scores)
    :param metrics_score_coa: List of metrics for classifying the presence score of each emotion
    :param metrics_presence: T lists of lists of metrics for detecting the presence/absence of an emotion (T is the number of thresholds set)
    :param metrics_dominant: List of metrics for detecting the dominant emotion(s)
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param extension_name: extension name containing info on whether we predict sentiment/neutral class
    :return: One-line results added to the csv file.
    """

    # Create filenames
    csv_path_regression = create_csv_path(model_name, filename="regression", extension_name=extension_name)
    csv_path_score_coa = create_csv_path(model_name, filename="classif_score_coa",
                                         extension_name=extension_name)
    csv_path_presence = [create_csv_path(model_name, filename="classif_pres_t_{}".format(thres),
                                         extension_name=extension_name) for thres in threshold_emo_pres]
    csv_path_dominant = create_csv_path(model_name, filename="classif_dom", extension_name=extension_name)

    # Create model parameter header and data for each csv file
    header_param = ['model_id', loss_function]
    data_param = [model_id, loss_function_val]

    # Regression metrics
    if metrics_regression is not None:
        header_regression, data_regression = get_header_and_data_metrics(metrics_regression, header_param, data_param,
                                                                         predict_neutral_class, task="regression")
        write_csv(csv_path_regression, header_regression, data_regression)

    # Coarse presence score metrics
    if metrics_score_coa is not None:
        header_score_coa, data_score_coa = get_header_and_data_metrics(metrics_score_coa, header_param, data_param,
                                                                       predict_neutral_class, task="score_coa")
        write_csv(csv_path_score_coa, header_score_coa, data_score_coa)

    # Present emotion metrics
    if metrics_presence is not None:
        header_presence, data_presence = get_header_and_data_metrics(metrics_presence, header_param, data_param,
                                                                     predict_neutral_class, task="presence")
        for path, data in zip(csv_path_presence, data_presence):
            write_csv(path, header_presence, data)

    # Dominant emotion metrics
    header_dominant, data_dominant = get_header_and_data_metrics(metrics_dominant, header_param, data_param,
                                                                 predict_neutral_class, task="dominant")
    write_csv(csv_path_dominant, header_dominant, data_dominant)

    # print(len(header_regression), len(data_regression))
    # print(len(header_score_coa), len(data_score_coa))
    # print(len(header_dominant), len(data_dominant))
    # for i in range(len(metrics_presence)):
    #     print(len(header_presence), len(data_presence[i]))


def save_model_param_in_csv_file(model_id, param_archi, param_training,
                                 predict_neutral_class, label_type, set_length, model_name):
    """
    Create a csv file which lists the model ids and model architecture/training parameters.

    :param model_id: Model id (int)
    :param param_archi: dictionary of 'architecture' hyperparameters of the model chosen (cf. hyperparameters.json)
    :param param_training: dictionary of 'training' hyperparameters of the model chosen (cf. hyperparameters.json)
    :param predict_neutral_class: Whether we predict the neutral class
    :param label_type: Type of labels used for model training and evaluation. Possible values: all (for all presence
                         scores provided by the dataset), coarse (presence score for each emotion among "
                         [0, 1, 2, 3]), present (multi-label classification), dominant (multi-label classification)
    :param set_length: Set length for all sequences in dataset (variable length, padding or truncate for fixed length)
    :param model_name: Name of the model
    """

    csv_path = create_csv_path(model_name, "model_ids")
    num_classes = 7 if predict_neutral_class else 6

    # Model architecture hyperparameters
    data_archi = list(param_archi.values())
    header_archi = list(param_archi.keys())
    data = [model_id, model_name] + data_archi + [num_classes]
    header = ['model_id', 'model_name'] + header_archi + ['num_classes']

    # Model training hyperparameters
    data_training = list(param_training.values())
    header_training = list(param_training.keys())
    data = data + data_training + [predict_neutral_class, label_type, set_length]
    header = header + header_training + ['neutral_class', 'label_type', 'set_length']

    write_csv(csv_path, header, data)
