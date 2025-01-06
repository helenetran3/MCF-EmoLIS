import os.path

import numpy as np
from matplotlib import pyplot as plt

from src.pickle_functions import save_with_pickle, load_from_pickle
from src.csv_functions import save_results_in_csv_file
from src.label_processing import compute_true_labels
from src.print_functions import print_mean_std_cross_validation

from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mutual_info_score
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay


def round_not_nan(macro_val, round_decimals):
    return np.round(macro_val, round_decimals) if not np.isnan(macro_val) else np.nan


def compute_best_thres_for_monitor_metric(true_classes, pred_raw, round_decimals, best_thres_per_emo=None):
    """
    Return an array of shape (num_classes,) giving the best threshold for each class
       + the corresponding validation metric from model prediction.
    Possible threshold values: [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 ,
       0.75, 0.8 , 0.85, 0.9 , 0.95]

    :param true_classes: binary array of shape (split_set size, 6)
    :param pred_raw: array of shape (split_set size, 6), values between 0 and 1
    :param round_decimals: number of decimals to be rounded for metrics
    :param best_thres_per_emo: lists of 6 best thresholds
    :return: best_thres_per_emo, f1_each_rounded
    """

    num_classes = true_classes.shape[1]

    if best_thres_per_emo is None:
        # Compute the best thresholds for each emotion class
        thresholds = np.arange(0.05, 1, 0.05)
        val_metrics = []

        for thres in thresholds:
            pred_thres = np.empty(pred_raw.shape)
            pred_thres[pred_raw > thres] = 1
            pred_thres[pred_raw <= thres] = 0
            # val_metric_thres = []
            # for cl_i in range(num_classes):
            #     pred_bin_i = pred_thres[:, cl_i]
            #     true_i = true_classes[:, cl_i]
            #     # val_metric_thres += [f1_score(true_i, pred_bin_i, average='macro', zero_division=1)]
            # #val_metrics += [val_metric_thres]
            val_metrics += [f1_score(true_classes, pred_thres, average=None, zero_division=1)]  # TODO Temporary: Valid metric = f1
            # print(np.round(thres, 2), " - ", val_metrics[-1])

        best_thres_per_emo_ind = np.argmax(np.array(val_metrics), axis=0)
        best_thres_per_emo = thresholds[best_thres_per_emo_ind]
        # print("best_thres_per_emo:", best_thres_per_emo, "\n")

    # Compute validation metric based on model prediction
    pred_classes = np.empty(pred_raw.shape)  # For initialization
    for cl_i in range(num_classes):
        pred_raw_i = pred_raw[:, cl_i]
        pred_bin_i = np.empty(pred_raw_i.shape)
        pred_bin_i[pred_raw_i > best_thres_per_emo[cl_i]] = 1
        pred_bin_i[pred_raw_i <= best_thres_per_emo[cl_i]] = 0
        pred_classes[:, cl_i] = pred_bin_i
    assert ((pred_classes == 0) | (pred_classes == 1)).all()  # Check if pred_classes is binary

    f1_each = f1_score(true_classes, pred_classes, average=None, zero_division=1)
    f1_each_rounded = [np.round(val * 100, round_decimals) for val in f1_each]

    return best_thres_per_emo, f1_each_rounded


def compute_accuracy_per_class(is_weighted, num_classes, with_cross_val, model_id, model_folder, task,
                               thres=None, fold_id=None, score_id=None):
    """
    Compute a list of accuracies per emotion.

    :param is_weighted: whether the accuracy is weighted or not
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param with_cross_val: boolean (cross_validation or not)
    :param model_id: model id (int)
    :param model_folder: the folder where the results are saved (str)
    :param thres: threshold at which emotions are considered to be present. Must be between 0 and 3
    :param fold_id: Cross-validation fold id (int)
    :param score_id: Index position of the presence score (int)
    :return: a list of rounded accuracies
    """

    def compute_acc_i(conf_mat_i):
        acc_list_i = []
        for emo_i in range(num_classes):
            tp = conf_mat_i[emo_i, 1, 1]
            fp = conf_mat_i[emo_i, 0, 1]
            fn = conf_mat_i[emo_i, 1, 0]
            tn = conf_mat_i[emo_i, 0, 0]
            neg = tn + fp
            pos = tp + fn
            if is_weighted:
                # print("tp:", tp, ", fp:", fp, ", fn:", fn, ", tn:", tn)
                # Formula from Tong, E., Zadeh, A., Jones, C., & Morency, L. P. (2017, July). Combating Human
                # Trafficking with Multimodal Deep Models. In Proceedings of the 55th Annual Meeting of the
                # Association for Computational Linguistics (Volume 1: Long Papers) (pp. 1547-1556).
                if pos == 0 or neg == 0:
                    acc_emo = np.nan
                else:
                    acc_emo = (tp * (neg / pos) + tn) / (2 * neg)
            else:
                acc_emo = (tp + tn) / (tp + fp + fn + tn)
            acc_list_i.append(acc_emo)
        return acc_list_i

    ext_name = task[:3] if thres is None else "pres_t_{}".format(thres)
    ext_name += "_cv" if with_cross_val else ""
    conf_matrix = load_from_pickle('conf_matrix_{}_{}'.format(model_id, ext_name),
                                   pickle_folder="confusion_matrix", root_folder=model_folder)
    conf_matrix_i = conf_matrix[fold_id] if fold_id is not None else conf_matrix
    conf_matrix_i = conf_matrix_i[score_id] if score_id is not None else conf_matrix_i
    acc_list = compute_acc_i(conf_matrix_i)
    weight_classes_for_mean = conf_matrix_i[:, 1, 0] + conf_matrix_i[:, 1, 1]

    return acc_list, weight_classes_for_mean


def get_regression_metrics(true_scores, pred_scores, label_type, all_scores, round_decimals):
    """
    Compute regression metrics: mean absolute error, mean squared error, r2 score and KL divergence + concordance index
    (if applicable)

    :param true_scores: array of shape (test size, 6) giving the true scores for the 6 emotions (given by the database)
    :param pred_scores: array of shape (test_size, 6) predicting the presence score of the 6 emotions
    :param label_type: Type of labels used for model training and evaluation
    :param all_scores: list of available presence scores
    :param round_decimals: number of decimals to be rounded for metrics
    :return: List of classification metrics
    """

    def concordance_index(t, p):
        scores = all_scores if label_type == 'all' else [0, 1, 2, 3]
        # Create list of boolean values where
        index_scores = [list(np.where(t == sc_i)[0]) for sc_i in scores]
        num_values = t.shape[0] * t.shape[1] if t.ndim == 2 else t.shape[0]
        assert(np.sum([len(ind) for ind in index_scores]) == num_values)

        res, total_values = 0., 0
        # i and j are indexes in scores
        for i in range(len(scores)-1):
            for j in range(1, len(scores)):
                if i < j:
                    index_inf, index_sup = index_scores[i], index_scores[j]
                    card_inf_times_card_sup = len(index_inf) * len(index_sup)
                    if card_inf_times_card_sup != 0:
                        total_values += card_inf_times_card_sup
                        for i_sup_sample in index_sup:
                            for i_inf_sample in index_inf:
                                if t.ndim == 2:
                                    i_sup_row, i_sup_col = i_sup_sample
                                    i_inf_row, i_inf_col = i_inf_sample
                                    p_val_sup, p_val_inf = p[i_sup_row, i_sup_col], p[i_inf_row, i_inf_col]
                                else:  # dimension of t is 1
                                    p_val_sup, p_val_inf = p[i_sup_sample], p[i_inf_sample]
                                if p_val_sup > p_val_inf:
                                    res += 1.
                                if p_val_sup == p_val_inf:
                                    res += 0.5

        return res / total_values

    def KL(t, p):
        return np.sum(np.where(t != 0, t * np.log(t / p), 0))

    num_emotions = true_scores.shape[1]
    true_scores_emo = [true_scores[:, i] for i in range(num_emotions)]
    pred_scores_emo = [pred_scores[:, i] for i in range(num_emotions)]

    mae_emo = [mean_absolute_error(t, p) for t, p in zip(true_scores_emo, pred_scores_emo)]
    mae = mean_absolute_error(true_scores, pred_scores)
    print("Mean absolute error (for each emotion):", [round_not_nan(val, round_decimals+2) for val in mae_emo])
    print("Mean absolute error (overall):", round_not_nan(mae, round_decimals))

    mse_emo = [mean_squared_error(t, p) for t, p in zip(true_scores_emo, pred_scores_emo)]
    mse = mean_squared_error(true_scores, pred_scores)
    print("Mean squared error (for each emotion):", [round_not_nan(val, round_decimals+2) for val in mse_emo])
    print("Mean squared error (overall):", round_not_nan(mse, round_decimals))

    r2_emo = [r2_score(t, p)*100 for t, p in zip(true_scores_emo, pred_scores_emo)]
    r2 = r2_score(true_scores, pred_scores)
    print("R2 score (for each emotion):", [round_not_nan(val, round_decimals) for val in r2_emo])
    print("R2 score (overall):", round_not_nan(r2, round_decimals))

    kl_div_emo = [KL(t, p) for t, p in zip(true_scores_emo, pred_scores_emo)]
    kl_div = KL(true_scores, pred_scores)
    print("KL divergence (for each emotion):", [round_not_nan(val, round_decimals+2) for val in kl_div_emo])
    print("KL divergence (overall):", round_not_nan(kl_div, round_decimals))

    c_index_emo, c_index = [None for _ in range(6)], None
    if label_type == 'all' or 'coarse' in label_type:
        c_index_emo = [concordance_index(t, p)*100 for t, p in zip(true_scores_emo, pred_scores_emo)]
        c_index = np.mean(c_index_emo)
        print("C-index (for each emotion):", [round_not_nan(val, round_decimals) for val in c_index_emo])
        print("C-index (unweighted mean):", round_not_nan(c_index, round_decimals))

    return [mae, mse, r2, kl_div, c_index] + mae_emo + mse_emo + r2_emo + kl_div_emo + c_index_emo


def get_classification_metrics_score(true_scores, pred_scores, num_classes, round_decimals,
                                     model_id, model_folder, with_cross_val, fold_id, coarse):
    """
    Return unweighted mean of all the classification metrics for the case of coarse scores.

    :param true_scores: Array of shape (test_size, 6) giving the true presence scores among [0, 1, 2, 3]
    :param pred_scores: Array of shape (test_size, 6) giving the predicted presence scores among [0, 1, 2, 3]
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param round_decimals: number of decimals to be rounded for metrics
    :param model_id: model id (int)
    :param model_folder: the folder where the results are saved (str)
    :param with_cross_val: boolean (cross_validation or not)
    :param fold_id: Cross-validation fold id (int)
    :param coarse: if True, the resulting presence scores in [0, 1, 2, 3], else [0, 0.33, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    :return: [f1_macro_uw_mean, f1_weighted_uw_mean, rec_macro_uw_mean, rec_weighted_uw_mean, prec_macro_uw_mean,
              prec_weighted_uw_mean, roc_auc_macro_uw_mean, roc_auc_weighted_uw_mean]
    """

    scores = [0, 1, 2, 3] if coarse else [0, 0.33, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    task = "coa" if coarse else "all"

    res = []
    for sc_id, sc_i in enumerate(scores):
        print("\n>>> Score threshold:", sc_i)
        true_classes_i = (true_scores > sc_i).astype(int)
        pred_classes_i = (pred_scores > sc_i).astype(int)
        res += [get_classification_metrics(true_classes_i, pred_classes_i, num_classes, round_decimals, task,
                                           model_id, model_folder, with_cross_val, fold_id=fold_id, score_id=sc_id)]

    return res


def get_classification_metrics(true_classes, pred_classes, num_classes, round_decimals, task,
                               model_id, model_folder, with_cross_val, fold_id=None, thres=None, score_id=None):
    """
    Compute classification metrics.
     Accuracy
     F1 for each emotion, unweighted and weighted F1
     Recall for each emotion, unweighted and weighted recall
     Precision for each emotion, unweighted and weighted precision
     ROC AUC for each emotion, unweighted and weighted ROC AUC
    :param true_classes: binary array of true classes, shape (test_size, num_classes) (num_classes = 7 if with neutral class, else 6)
    :param pred_classes: binary array of predictions, shape (test_size, num_classes) (num_classes = 7 if with neutral class, else 6)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param round_decimals: number of decimals to be rounded for metrics
    :param task: Name of the task, among 'all', 'coa', 'pres' and 'dom'
    :param model_id: Model id (int)
    :param model_folder: The folder where the result will be saved
    :param with_cross_val: boolean (cross_validation or not)
    :param fold_id: Cross-validation fold id (int)
    :param thres: threshold at which emotions are considered to be present. Must be between 0 and 3 (int)
    :param score_id: Index position of the presence score (int)
    :return: List of classification metrics
    """

    if thres is not None:
        print('\n>>> Threshold:', thres)

    # Accuracy
    bin_acc_per_class, weight_classes_for_mean = \
        compute_accuracy_per_class(False, num_classes, with_cross_val, model_id, model_folder,
                                   task, thres=thres, fold_id=fold_id, score_id=score_id)
    wei_acc_per_class, _ = compute_accuracy_per_class(True, num_classes, with_cross_val, model_id, model_folder,
                                                      task, thres=thres, fold_id=fold_id, score_id=score_id)
    bin_acc_per_class = [bin_acc_i*100 for bin_acc_i in bin_acc_per_class]
    wei_acc_per_class = [wei_acc_i*100 if not np.isnan(wei_acc_i) else np.nan for wei_acc_i in wei_acc_per_class]
    multilabel_acc = accuracy_score(true_classes, pred_classes)*100  # True and pred class vectors must exactly match
    bin_acc_macro = np.mean(bin_acc_per_class)
    wei_acc_macro = np.mean([w for w in wei_acc_per_class if not np.isnan(w)])
    bin_acc_weighted = np.average(bin_acc_per_class, weights=weight_classes_for_mean)
    index_not_nan = [i for i, w in enumerate(wei_acc_per_class) if not np.isnan(w)]
    if len(index_not_nan) != 0:
        wei_acc_weighted = np.average([wei_acc_per_class[i] for i in index_not_nan],
                                      weights=[weight_classes_for_mean[i] for i in index_not_nan])
    else:
        wei_acc_weighted = np.nan
    print("Multilabel accuracy:", round_not_nan(multilabel_acc, round_decimals))
    print("Binary accuracy (for each):", [round_not_nan(bin_acc_i, round_decimals) for bin_acc_i in bin_acc_per_class])
    print("Binary accuracy (unweighted mean):", round_not_nan(bin_acc_macro, round_decimals))
    print("Binary accuracy (weighted mean):", round_not_nan(bin_acc_weighted, round_decimals))
    print("Weighted accuracy (for each):", [round_not_nan(wei_acc_i, round_decimals) for wei_acc_i in wei_acc_per_class])
    print("Weighted accuracy (unweighted mean):", round_not_nan(wei_acc_macro, round_decimals))
    print("Weighted accuracy (weighted mean):", round_not_nan(wei_acc_weighted, round_decimals))

    # F1 score
    f1_each = [val*100 for val in f1_score(true_classes, pred_classes, average=None, zero_division=1)]
    f1_macro = f1_score(true_classes, pred_classes, average='macro', zero_division=1)*100
    f1_weighted = f1_score(true_classes, pred_classes, average='weighted', zero_division=1)*100
    print("F1 score (for each):", [round_not_nan(f1_i, round_decimals) for f1_i in f1_each])
    print("F1 score (unweighted mean):", round_not_nan(f1_macro, round_decimals))
    print("F1 score (weighted mean):", round_not_nan(f1_weighted, round_decimals))

    # Recall
    rec_each = [val*100 for val in recall_score(true_classes, pred_classes, average=None, zero_division=1)]
    rec_macro = recall_score(true_classes, pred_classes, average='macro', zero_division=1)*100
    rec_weighted = recall_score(true_classes, pred_classes, average='weighted', zero_division=1)*100
    print("Recall (for each):", [round_not_nan(rec_i, round_decimals) for rec_i in rec_each])
    print("Recall (unweighted mean):", round_not_nan(rec_macro, round_decimals))
    print("Recall (weighted mean):", round_not_nan(rec_weighted, round_decimals))

    # Precision
    prec_each = [val*100 for val in precision_score(true_classes, pred_classes, average=None, zero_division=1)]
    prec_macro = precision_score(true_classes, pred_classes, average='macro', zero_division=1)*100
    prec_weighted = precision_score(true_classes, pred_classes, average='weighted', zero_division=1)*100
    print("Precision (for each):", [round_not_nan(prec_i, round_decimals) for prec_i in prec_each])
    print("Precision (unweighted mean):", round_not_nan(prec_macro, round_decimals))
    print("Precision (weighted mean):", round_not_nan(prec_weighted, round_decimals))

    # Jaccard score
    jacc_each = [val*100 for val in jaccard_score(true_classes, pred_classes, average=None, zero_division=1)]
    jacc_macro = jaccard_score(true_classes, pred_classes, average='macro', zero_division=1)*100
    jacc_weighted = jaccard_score(true_classes, pred_classes, average='weighted', zero_division=1)*100
    print("Jaccard score (for each):", [round_not_nan(jacc_i, round_decimals) for jacc_i in jacc_each])
    print("Jaccard score (unweighted mean):", round_not_nan(jacc_macro, round_decimals))
    print("Jaccard score (weighted mean):", round_not_nan(jacc_weighted, round_decimals))

    # # ROC AUC score
    # true_classes_bin = label_binarize(true_classes, classes=list(range(num_classes)))
    # pred_classes_bin = label_binarize(pred_classes, classes=list(range(num_classes)))
    # roc_auc_each = [val*100 for val in roc_auc_score(true_classes_bin, pred_classes_bin, average=None)]
    # roc_auc_macro = roc_auc_score(true_classes_bin, pred_classes_bin, average='macro')*100
    # roc_auc_weighted = roc_auc_score(true_classes_bin, pred_classes_bin, average='weighted', multi_class='ovr')*100
    # print("ROC AUC (for each):", [round_not_nan(roc_auc_i, round_decimals) for roc_auc_i in roc_auc_each])
    # print("ROC AUC (unweighted mean):", round_not_nan(roc_auc_macro, round_decimals))
    # print("ROC AUC (weighted mean):", round_not_nan(roc_auc_weighted, round_decimals))

    res = [multilabel_acc, bin_acc_macro, bin_acc_weighted, wei_acc_macro, wei_acc_weighted, f1_macro, f1_weighted,
           rec_macro, rec_weighted, prec_macro, prec_weighted, jacc_macro, jacc_weighted]  # , roc_auc_macro, roc_aux_weighted]
    res = res + bin_acc_per_class + wei_acc_per_class + f1_each + rec_each + prec_each + jacc_each  # + roc_auc_each

    return res


def plot_multilabel_confusion_matrix(num_classes, with_cross_val, num_folds, model_id, model_folder, label_type,
                                     display_fig, thres_score_pres):
    """
    Generate up to three plots, depending on the label type: one for fine-grained presence score, one for coarse-grained,
    and one for dominant emotions (and present if applicable).
    :param num_classes: number of classes to predict (int)
    :param with_cross_val: whether cross-validation is performed or not (bool)
    :param num_folds: number of folds in cross-validation (int)
    :param model_id: model id (int)
    :param model_folder: the folder where the results are saved (str)
    :param label_type: type of labels used for model training and evaluation ('all', 'coarse', 'coarse_prob',
    'coarse_oh', 'present', 'dominant')
    :param display_fig: whether we display the figures (bool)
    :param thres_score_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    """
    display = None
    end_of_title_dict = {'all': " fine-grained",
                         'coa': " coarse-grained",
                         'pres': " with score>",
                         'dom': " dominant"}

    emotions = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    if num_classes == 7:
        emotions += ['neutral']
    ext_name = "_cv" if with_cross_val else ""

    def plot_task(task_name, score_batch=None, score_batch_id=None):

        def plot_one_confusion_matrix(conf_matrix, emo_i, axe_id, end_of_title, score_id=None):
            conf_matrix_i = conf_matrix[score_id][emo_i, :, :] if score_id is not None else conf_matrix[emo_i, :, :]
            d = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_i, display_labels=[0, 1])
            d.plot(ax=axes[axe_id], values_format='.4g')
            d.ax_.set_title(emotions[emo_i] + end_of_title)
            # if task_name == 'all':  # 90° rotation of x labels for better readability
            #     d.ax_.tick_params(axis='x', rotation=90)
            if emo_i != 0:
                d.ax_.set_ylabel('')
            d.im_.colorbar.remove()

            return d

        def cross_valid_plot_matrix(conf_matrix, line_id, task_name, thres=None):
            conf_mean = np.mean(conf_matrix, axis=0)
            end_of_title = end_of_title_dict[task_name] if thres is None else end_of_title_dict[task_name] + str(thres)
            end_of_title = end_of_title + " (mean of {} folds)".format(num_folds)
            for emo_i in range(num_classes):
                d = plot_one_confusion_matrix(conf_mean, emo_i, line_id * num_classes + emo_i, end_of_title)
            return d

        def hold_out_plot_matrix(conf_matrix, line_id, task_name, thres=None):
            end_of_title = end_of_title_dict[task_name] if thres is None else end_of_title_dict[task_name] + str(thres)
            for emo_i in range(num_classes):
                d = plot_one_confusion_matrix(conf_matrix, emo_i, line_id * num_classes + emo_i, end_of_title)
            return d

        # Plot confusion matrix
        if task_name == 'all' or task_name == 'coa':
            f, axes = plt.subplots(4, num_classes, figsize=(3 * num_classes, 3 * 4), tight_layout=True)
            axes = axes.ravel()
            conf_matrix = load_from_pickle('conf_matrix_{}_{}{}'.format(model_id, task_name, ext_name),
                                           pickle_folder="confusion_matrix", root_folder=model_folder)
            # Below, score_batch is [k, k + 0.33, k + 0.66, k + 1] for 'all' or [0, 1, 2, 3] for 'coa'
            for ii, sc_i in enumerate(score_batch):
                sc_index = 0 + ii + 3 * score_batch_id if task_name == 'all' else sc_i
                if with_cross_val:
                    conf_matrix_for_score_i = [conf_matrix[fold_i][sc_index] for fold_i in range(num_folds)]
                    display = cross_valid_plot_matrix(np.array(conf_matrix_for_score_i), line_id=ii, task_name=task_name)
                else:
                    display = hold_out_plot_matrix(conf_matrix[sc_index], line_id=ii, task_name=task_name)

        elif task_name == 'pres':
            f, axes = plt.subplots(len(thres_score_pres), num_classes,
                                   figsize=(3 * num_classes, 3 * len(thres_score_pres)), tight_layout=True)
            axes = axes.ravel()
            for i, thres in enumerate(thres_score_pres):
                conf_matrix_pres = load_from_pickle('conf_matrix_{}_pres_t_{}{}'.format(model_id, thres, ext_name),
                                                    pickle_folder="confusion_matrix", root_folder=model_folder)
                if with_cross_val:
                    display = cross_valid_plot_matrix(conf_matrix_pres, line_id=i, thres=thres, task_name='pres')
                else:
                    display = hold_out_plot_matrix(conf_matrix_pres, line_id=i, thres=thres, task_name='pres')

        else:  # task_name = dom
            f, axes = plt.subplots(1, num_classes, figsize=(3 * num_classes, 3))
            axes = axes.ravel()
            conf_matrix = load_from_pickle('conf_matrix_{}_{}{}'.format(model_id, task_name, ext_name),
                                           pickle_folder="confusion_matrix", root_folder=model_folder)
            if with_cross_val:
                display = cross_valid_plot_matrix(conf_matrix, line_id=0, task_name=task_name)
            else:
                display = hold_out_plot_matrix(conf_matrix, line_id=0, task_name=task_name)

        # Save figure
        score_batch_str = "_{}".format(score_batch_id) if score_batch_id is not None else ""
        plt.savefig(os.path.join(model_folder, "confusion_matrix",
                                 "conf_matrix_{}_{}{}.png".format(model_id, task_name, score_batch_str)),
                    bbox_inches='tight')

        f.colorbar(display.im_, ax=axes)
        if display_fig:
            plt.show()

    if label_type == 'all':
        scores = [[k, k + 0.33, k + 0.66, k + 1] for k in [0, 1, 2]]
        for score_batch_id, score_batch in enumerate(scores):
            # Generate three plots: [0, 0.33, 0.66, 1], [1, 1.33, 1.66, 2], [2, 2.33, 2.66, 3]
            plot_task("all", score_batch, score_batch_id)
    if 'coarse' in label_type:
        plot_task("coa", score_batch=[0, 1, 2, 3])
    if label_type != 'dominant':
        plot_task("pres")
    if label_type != 'present':
        plot_task("dom")


def compute_multilabel_confusion_matrix(true_classes, pred_classes, with_cross_val, num_classes,
                                        model_id, model_folder, task_name, scores=None, thres_score_pres=None):
    """
    Compute the multilabel confusion matrix for each emotion (whether the emotion is present or not).
    Pickle file generated.

    :param true_classes: List of arrays giving the emotions truly present (varying thresholds for presence score
    given by thres_score_pres) or the dominant emotions
    :param pred_classes: List of arrays giving the emotions present/dominant predicted by the model
    :param with_cross_val: boolean (cross_validation or not)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param model_id: Model id (int)
    :param model_folder: The folder where the results will be saved
    :param task_name: Name of the task, among 'all', 'coa', 'pres' and 'dom'
    :param scores: list of scores threshold used for binarizing
    :param thres_score_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3.
    This parameter should be used only when dealing with emotions present.
    """

    def compute_all_conf_matrices(task_n, thres=None, thres_id=None):

        def compute_conf_matrix(tc, pc):
            if scores is not None:
                conf_mat = []
                for sc_i in scores:
                    tc_i, pc_i = (tc >= sc_i).astype(int), (pc >= sc_i).astype(int)
                    # print("[conf matrix] sc_i:", sc_i)
                    # print("[conf matrix] tc_i unique:", np.unique(tc_i, return_counts=True))
                    conf_mat += [multilabel_confusion_matrix(tc_i, pc_i, labels=list(range(num_classes)))]
            else:
                conf_mat = multilabel_confusion_matrix(tc, pc, labels=list(range(num_classes)))
            return conf_mat

        task_n = "pres_t_{}".format(thres) if thres is not None else task_n
        cv_str = "_cv" if with_cross_val else ""
        if with_cross_val:
            conf_matrix = []
            num_folds = len(pred_classes)  # pred_classes is a list of num_fold elements
            for fold_id in range(num_folds):
                true_c = true_classes[fold_id][thres_id] if thres_id is not None else true_classes[fold_id]
                pred_c = pred_classes[fold_id][thres_id] if thres_id is not None else pred_classes[fold_id]
                conf_matrix += [compute_conf_matrix(true_c, pred_c)]
        else:
            true_c = true_classes[thres_id] if thres_id is not None else true_classes
            pred_c = pred_classes[thres_id] if thres_id is not None else pred_classes
            conf_matrix = compute_conf_matrix(true_c, pred_c)
        save_with_pickle(conf_matrix, 'conf_matrix_{}_{}{}'.format(model_id, task_n, cv_str),
                         pickle_folder="confusion_matrix", root_folder=model_folder)

    # emotions present
    if task_name == 'pres':
        for thres_id, thres in enumerate(thres_score_pres):
            compute_all_conf_matrices(task_name, thres, thres_id)

    # emotions dominant or emotions evaluated on presence scores
    elif task_name == 'dom' or task_name == 'coa' or task_name == 'all':
        compute_all_conf_matrices(task_name)

    else:
        raise ValueError("Error in task_name parameter of function compute_multilabel_confusion_matrix: "
                         "please choose between 'all', 'coa', 'pres' and 'dom'.")


def get_and_print_all_metrics(label_type, true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom,
                              pred_raw, pred_classes_pres, pred_classes_dom, with_cross_val, thres_score_pres,
                              num_classes, all_scores, round_decimals, model_id, model_folder):
    """
    Compute metrics for four tasks: estimation of presence score (regression), classification of presence score among
    [0, 1, 2, 3], binary classification for each emotion, and prediction of dominant classes (multilabel classification)

    :param label_type: Type of labels used for model training and evaluation.
    :param true_scores_all: array of shape (test_size, num_classes) giving the true presence score of the emotions
    :param true_scores_coa: array of shape (test_size, num_classes) giving the true presence score among [0, 1, 2, 3]
    :param true_classes_pres: list of binary arrays of shape (test_size, num_classes) giving the true presence of an
    emotion (> threshold defined by thres_score_pres)
    :param true_classes_dom: binary array of shape (test_size, num_classes) giving the truly dominant emotions
    :param pred_raw: array of shape (test_size, num_classes) from the model raw output
    :param pred_classes_pres: list of binary arrays of shape (test_size, num_classes) giving the predicted presence of
    an emotion (> threshold defined by thres_score_pres)
    :param pred_classes_dom: binary array of shape (test_size, num_classes) giving the predicted dominant emotions
    :param with_cross_val: boolean (cross_validation or not)
    :param all_scores: list of available presence scores
    :param thres_score_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param round_decimals: Number of decimals to be rounded for metrics
    :param model_id: Model id (int)
    :param model_folder: The folder where the result will be saved
    :return: metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant
    """

    # Some metrics do not make sense for the targeted task (label_type): we do not print them
    label_type_for_reg = 'coarse' if 'coarse' in label_type else label_type

    def compute_metrics_one_model(true_all_i, true_coa_i, true_pres_i, true_dom_i,
                                  pred_raw_i, pred_pres_i, pred_dom_i, fold_id=None):

        true_label_for_reg = {'all': true_all_i, 'coarse': true_coa_i, 'present': true_pres_i, 'dominant': true_dom_i}
        metrics_score, metrics_presence, metrics_dominant = None, None, None

        if fold_id is not None:
            print("\n\n=============================== Fold {} ===============================".format(fold_id))

        # This block computes regression metrics for the targeted task
        print("\n------- Presence score estimation (regression) --------\n")
        tl_for_reg = true_label_for_reg[label_type_for_reg]
        tl_for_reg = tl_for_reg[0] if label_type == 'present' else tl_for_reg
        metrics_regression = get_regression_metrics(tl_for_reg, pred_raw_i, label_type, all_scores, round_decimals)

        if 'coarse' in label_type or label_type == 'all':
            print(f"\n------- Presence score classification -------\n")
            coarse = ('coarse' in label_type)
            metrics_score = get_classification_metrics_score(true_all_i, pred_raw_i, num_classes, round_decimals,
                                                             model_id, model_folder, with_cross_val, fold_id, coarse)

        # Classification metrics of emotion present
        if pred_pres_i is not None:
            print("\n------- Detecting the presence of emotions ------------")
            metrics_presence = [get_classification_metrics(true_pres_i[j], pred_pres_i[j], num_classes, round_decimals,
                                                           "pres", model_id, model_folder, with_cross_val,
                                                           fold_id=fold_id, thres=thres)
                                for j, thres in enumerate(thres_score_pres)]

        # Classification metrics of dominant emotions
        if pred_dom_i is not None:
            print("\n------- Predicting dominant emotions ------------------\n")
            metrics_dominant = get_classification_metrics(true_dom_i, pred_dom_i, num_classes, round_decimals, "dom",
                                                          model_id, model_folder, with_cross_val, fold_id=fold_id)

        return metrics_regression, metrics_score, metrics_presence, metrics_dominant

    if with_cross_val:
        met_reg, met_score, met_pres, met_dom = [], [], [], []
        for fold_id, (tf_i, tc_i, tp_i, td_i, pr_i, pp_i, pd_i) in \
                enumerate(zip(true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom,
                              pred_raw, pred_classes_pres, pred_classes_dom)):
            mr, ms, mp, md = compute_metrics_one_model(tf_i, tc_i, tp_i, td_i, pr_i, pp_i, pd_i, fold_id)
            met_reg += [mr]
            met_score += [ms]
            met_pres += [mp]
            met_dom += [md]

        # Print Mean and Std of values
        print_mean_std_cross_validation(met_reg, met_score, met_pres, met_dom, label_type, num_classes,
                                        thres_score_pres, round_decimals)

    else:
        met_reg, met_score, met_pres, met_dom = compute_metrics_one_model(true_scores_all, true_scores_coa,
                                                                          true_classes_pres, true_classes_dom,
                                                                          pred_raw, pred_classes_pres, pred_classes_dom)
    return met_reg, met_score, met_pres, met_dom


def compute_and_save_model_metrics(test_list_task, with_cross_val, model_eval_output,
                                   pred_raw, pred_classes_pres, pred_classes_dom,
                                   param_task, param_training, param_log, model_name, model_id,
                                   loss_function, val_metrics_str, all_scores):
    """
    Compute and save model metrics, based on loss function value and prediction scores.

    :param test_list_task: [x_test, y_test, seg_test] from original dataset
    :param with_cross_val: boolean (cross_validation or not)
    :param model_eval_output: List of [loss function, validation metric] obtained by the model
    :param pred_raw: array of shape (test_size, num_classes) predicting the presence score of the emotions
    :param pred_classes_pres: list of binary arrays of shape (test_size, num_classes) giving the predicted presence of
    an emotion (> threshold defined by thres_score_pres)
    :param pred_classes_dom: binary array of shape (test_size, num_classes) giving the predicted dominant emotions
    :param param_log: dictionary of 'logistics' hyperparameters (cf. param_logistics.json)
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param param_training: dictionary of 'training' hyperparameters (cf. param_training_task.json)
    :param model_name: Name of the model currently tested
    :param model_id: Model id (int)
    :param loss_function: Name of the loss function (str)
    :param val_metrics_str: Name of the validation metric used for monitoring (str)
    :param all_scores: list of available presence scores [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    """

    predict_neutral_class = param_task['predict_neutral_class']
    thres_score_pres = [param_task['thres_score_pres']]
    label_type = param_task['label_type']
    round_decimals = param_log['round_decimals']
    plot_confusion_matrix = param_log['plot_confusion_matrix']
    display_fig = param_log['display_fig']
    num_folds = param_training['cross_val_folds']

    model_folder = os.path.join('results', model_name)

    # Compute true labels for calculation metrics on test set
    def compute_true_label_for_metrics(y_test):
        true_all, true_coa, true_pres, true_dom = compute_true_labels(y_test, "test", all_scores, predict_neutral_class,
                                                                      thres_score_pres, param_task, num_classes,
                                                                      label_type, for_test=True)

        return true_all, true_coa, true_pres, true_dom

    if with_cross_val:
        y_test_cv_fold_0 = test_list_task[0][1]
        num_classes = y_test_cv_fold_0[0].shape[1] if not ('oh' in label_type) else y_test_cv_fold_0[0].shape[0]
        true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom = [], [], [], []
        for fold_i in range(num_folds):
            ts_all, ts_coa, tc_pres, tc_dom = compute_true_label_for_metrics(test_list_task[fold_i][1])
            true_scores_all += [ts_all]
            true_scores_coa += [ts_coa]
            true_classes_pres += [tc_pres]
            true_classes_dom += [tc_dom]
    else:
        y_test_hold_out = test_list_task[1]
        num_classes = y_test_hold_out[0].shape[1] if not ('oh' in label_type) else y_test_hold_out[0].shape[0]
        true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom = \
            compute_true_label_for_metrics(test_list_task[1])

    # Confusion matrices
    # if label_type == 'all':
    #     compute_multilabel_confusion_matrix(true_scores_all, pred_scores_all, with_cross_val, num_classes,
    #                                         model_id, model_folder, "all", all_scores)
    #
    if label_type != 'dominant' and label_type != 'present':
        true_scores = true_scores_all if label_type == 'all' else true_scores_coa
        scores = [0, 0.33, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3] if label_type == 'all' else [0, 1, 2, 3]
        compute_multilabel_confusion_matrix(true_scores, pred_raw, with_cross_val, num_classes,
                                            model_id, model_folder, label_type[:3], scores=scores)
    if label_type != 'dominant':
        compute_multilabel_confusion_matrix(true_classes_pres, pred_classes_pres, with_cross_val, num_classes,
                                            model_id, model_folder, "pres", thres_score_pres=thres_score_pres)
    if label_type != 'present':
        compute_multilabel_confusion_matrix(true_classes_dom, pred_classes_dom, with_cross_val, num_classes,
                                            model_id, model_folder, "dom")
    if plot_confusion_matrix:
        plot_multilabel_confusion_matrix(num_classes, with_cross_val, num_folds, model_id, model_folder, label_type,
                                         display_fig, thres_score_pres)

    print("\n\n")
    print("In the following, all the metrics displayed for each emotion are in this order: happiness, sadness, "
          "anger, surprise, disgust, fear", end="")
    print(", neutral.\n") if predict_neutral_class else print(".\n")

    print("\n----------- Model training characteristics ------------\n")
    print("Model name:", model_name)
    print("Model id:", model_id)
    print("Label type:", param_task['label_type'])
    print("Loss function:", loss_function)
    print("Validation metrics:", val_metrics_str)

    # Compute all metrics
    metrics_regression, metrics_score, metrics_presence, metrics_dominant = \
        get_and_print_all_metrics(label_type, true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom,
                                  pred_raw, pred_classes_pres, pred_classes_dom, with_cross_val, thres_score_pres,
                                  num_classes, all_scores, round_decimals, model_id, model_folder)

    # Save metrics in csv file
    save_results_in_csv_file(model_name, model_id, loss_function, val_metrics_str, model_eval_output,
                             metrics_regression, metrics_score, metrics_presence, metrics_dominant,
                             with_cross_val, predict_neutral_class, thres_score_pres, all_scores, label_type)
