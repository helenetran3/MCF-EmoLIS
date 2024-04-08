import os.path

import numpy as np
from matplotlib import pyplot as plt

from src.pickle_functions import save_with_pickle, load_from_pickle
from src.csv_functions import save_results_in_csv_file
from src.label_processing import compute_true_labels

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay


def get_regression_metrics(true_scores_all, pred_scores_all, round_decimals):
    """
    Compute regression metrics: mean absolute error and mean squared error

    :param true_scores_all: array of shape (test size, 6) giving the true scores for the 6 emotions (given by the database)
    :param pred_scores_all: array of shape (test_size, 6) predicting the presence score of the 6 emotions
    :param round_decimals: number of decimals to be rounded for metrics
    :return: List of classification metrics
    """

    num_emotions = true_scores_all.shape[1]
    true_scores_emo = [true_scores_all[:, i] for i in range(num_emotions)]
    pred_scores_emo = [pred_scores_all[:, i] for i in range(num_emotions)]

    mae_emo = [np.round(mean_absolute_error(t, p), round_decimals) for t, p in zip(true_scores_emo, pred_scores_emo)]
    mse_emo = [np.round(mean_squared_error(t, p), round_decimals) for t, p in zip(true_scores_emo, pred_scores_emo)]
    r2_emo = [np.round(r2_score(t, p), round_decimals) for t, p in zip(true_scores_emo, pred_scores_emo)]
    mae = np.round(mean_absolute_error(true_scores_all, pred_scores_all), round_decimals)
    mse = np.round(mean_squared_error(true_scores_all, pred_scores_all), round_decimals)
    r2 = np.round(r2_score(true_scores_all, pred_scores_all), round_decimals)

    print("Mean absolute error (for each emotion):", mae_emo)
    print("Mean squared error (for each emotion):", mse_emo)
    print("R2 score (for each emotion):", r2_emo)
    print("Mean absolute error (overall):", mae)
    print("Mean squared error (overall):", mse)
    print("R2 score (overall):", r2)

    return [mae, mse, r2] + mae_emo + mse_emo + r2_emo


def get_classification_metrics_score_coa(true_scores_coa, pred_scores_coa, num_classes, round_decimals, prob):
    """
    Return unweighted mean of all the classification metrics for the case of coarse scores.

    :param true_scores_coa: Array of shape (test_size, 6) giving the true presence scores among [0, 1, 2, 3]
    :param pred_scores_coa: Array of shape (test_size, 6) giving the predicted presence scores among [0, 1, 2, 3]
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param round_decimals: number of decimals to be rounded for metrics
    :param prob: If we use probabilities for coarse scores [0, 1/3, 2/3, 1]
    :return: [f1_macro_uw_mean, f1_weighted_uw_mean, rec_macro_uw_mean, rec_weighted_uw_mean, prec_macro_uw_mean,
              prec_weighted_uw_mean, roc_auc_macro_uw_mean, roc_auc_weighted_uw_mean]
    """

    # def get_binary_arrays_from_scores_coa(scores_coa):
    #     """
    #     Returns a list of num_classes binary arrays for each emotion. Each array has shape (test_size, 4).
    #     Useful for classification metrics from coarse presence scores.
    #
    #     :param scores_coa: array of size (test_size, num_classes) giving the coarse presence scores (0, 1, 2, or 3)
    #     :return: The list of binary arrays
    #     """
    #
    #     scores_for_each_emo = [scores_coa[:, i] for i in range(num_classes)]
    #     bin_arr_for_each_emo = [label_binarize(sc, classes=range(4)) for sc in scores_for_each_emo]
    #
    #     return bin_arr_for_each_emo
    #
    # # Lists of num_classes sublists of test_size elements (e.g. [1, 0, 0, 0])
    # true_scores_bin = get_binary_arrays_from_scores_coa(true_scores_coa)
    # pred_scores_bin = get_binary_arrays_from_scores_coa(pred_scores_coa)
    # # # Transform these lists of lists to arrays
    # # true_scores_bin_stack = np.vstack(true_scores_bin)
    # # pred_scores_bin_stack = np.vstack(pred_scores_bin)

    print("4 classes ([0,1,2,3] corresponding to presence scores) for each emotion class.\n")

    true_scores_for_each_emo = [true_scores_coa[:, i] for i in range(num_classes)]
    pred_scores_for_each_emo = [pred_scores_coa[:, i] for i in range(num_classes)]

    # If working with probabilities, go back to integers between 0, 1, 2, 3 to avoid metric miscalculation due to
    # floating points
    if prob:
        true_scores_for_each_emo = [int(ts*3) for ts in true_scores_for_each_emo]
        pred_scores_for_each_emo = [int(ts*3) for ts in true_scores_for_each_emo]

    acc_each = [np.round(accuracy_score(ts, ps), round_decimals)
                for ts, ps in zip(true_scores_for_each_emo, pred_scores_for_each_emo)]
    acc_macro = np.round(np.mean(acc_each), round_decimals)
    print("Accuracy (for each):", acc_each)
    print("Accuracy (unweighted mean):", acc_macro)

    # Only macro because with weighted, the over_presence of class "0" will bias the result
    f1_each = [np.round(f1_score(ts, ps, average='macro', zero_division=1), round_decimals)
               for ts, ps in zip(true_scores_for_each_emo, pred_scores_for_each_emo)]
    f1_macro = np.round(np.mean(f1_each), round_decimals)
    print("F1 score (for each):", f1_each)
    print("F1 score (unweighted mean):", f1_macro)

    rec_each = [np.round(recall_score(ts, ps, average='macro', zero_division=1), round_decimals)
                for ts, ps in zip(true_scores_for_each_emo, pred_scores_for_each_emo)]
    rec_macro = np.round(np.mean(rec_each), round_decimals)
    print("Recall (for each):", rec_each)
    print("Recall (unweighted mean):", rec_macro)

    prec_each = [np.round(precision_score(ts, ps, average='macro', zero_division=1), round_decimals)
                 for ts, ps in zip(true_scores_for_each_emo, pred_scores_for_each_emo)]
    prec_macro = np.round(np.mean(prec_each), round_decimals)
    print("Precision (for each):", prec_each)
    print("Precision (unweighted mean):", prec_macro)

    # roc_auc_each = [np.round(roc_auc_score(ts, ps, average='macro', multi_class='ovr'), round_decimals)
    #                 for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    # roc_auc_macro = np.round(np.mean(roc_auc_each), round_decimals)
    # print("ROC AUC (for each):", roc_auc_each)
    # print("ROC AUC (unweighted mean):", roc_auc_macro)

    # # In this block, we go through all the emotions. Metrics are here given for each emotion.
    # # Imbalance in presence score [0, 1, 2, 3]. example for happy: [3048, 1210, 376, 20],
    # # hence we also calculate weighted metrics
    # f1_macro_each = [np.round(f1_score(ts, ps, average='macro', zero_division=1), round_decimals)
    #                  for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    # f1_weighted_each = [np.round(f1_score(ts, ps, average='weighted', zero_division=1), round_decimals)
    #                     for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    # rec_macro_each = [np.round(recall_score(ts, ps, average='macro', zero_division=1), round_decimals)
    #                   for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    # rec_weighted_each = [np.round(recall_score(ts, ps, average='weighted', zero_division=1), round_decimals)
    #                      for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    # prec_macro_each = [np.round(precision_score(ts, ps, average='macro', zero_division=1), round_decimals)
    #                    for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    # prec_weighted_each = [np.round(precision_score(ts, ps, average='weighted', zero_division=1), round_decimals)
    #                       for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    # # roc_auc_macro_each = [roc_auc_score(ts, ps, average='macro')
    # #                      for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    # # roc_auc_weighted_each = [roc_auc_score(ts, ps, average='weighted')
    # #                         for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    #
    # # Unweighted mean over the six emotions
    # acc = np.round(accuracy_score(true_scores_bin_stack, pred_scores_bin_stack), round_decimals)
    # f1_macro_uw_mean = np.round(np.mean(f1_macro_each), round_decimals)
    # f1_weighted_uw_mean = np.round(np.mean(f1_weighted_each), round_decimals)
    # rec_macro_uw_mean = np.round(np.mean(rec_macro_each), round_decimals)
    # rec_weighted_uw_mean = np.round(np.mean(rec_weighted_each), round_decimals)
    # prec_macro_uw_mean = np.round(np.mean(prec_macro_each), round_decimals)
    # prec_weighted_uw_mean = np.round(np.mean(prec_weighted_each), round_decimals)
    # # roc_auc_macro_uw_mean = np.round(np.mean(roc_auc_macro_each), round_decimals)
    # # roc_auc_weighted_uw_mean = np.round(np.mean(roc_auc_weighted_each), round_decimals)
    # print("4 classes (presence scores) for each emotion class. "
    #       "The following metrics all give the unweighted mean over the emotions and the brackets give the type "
    #       "of mean over the presence scores.\n")
    # print("Accuracy:", acc)
    # print("Unweighted F1 score (for each emotion):", f1_macro_each)
    # print("Weighted F1 score (for each emotion):", f1_weighted_each)
    # print("Unweighted recall score (for each emotion):", rec_macro_each)
    # print("Weighted recall score (for each emotion):", rec_weighted_each)
    # print("Unweighted precision score (for each emotion):", prec_macro_each)
    # print("Weighted precision score (for each emotion):", prec_weighted_each)
    # print("F1 score (unweighted mean):", f1_macro_uw_mean)
    # print("F1 score (weighted mean):", f1_weighted_uw_mean)
    # print("Recall (unweighted mean):", rec_macro_uw_mean)
    # print("Recall (weighted mean):", rec_weighted_uw_mean)
    # print("Precision (unweighted mean):", prec_macro_uw_mean)
    # print("Precision (weighted mean):", prec_weighted_uw_mean)
    # # print("ROC AUC (unweighted mean):", roc_auc_macro_uw_mean)
    # # print("ROC AUC (weighted mean):", roc_auc_weighted_uw_mean)

    res = [acc_macro, f1_macro, rec_macro, prec_macro]  #, roc_auc_macro]
    res = res + acc_each + f1_each + rec_each + prec_each  #, roc_auc_each

    # res = [acc, f1_macro_uw_mean, f1_weighted_uw_mean, rec_macro_uw_mean, rec_weighted_uw_mean, prec_macro_uw_mean,
    #        prec_weighted_uw_mean]  # , roc_auc_macro_uw_mean, roc_auc_weighted_uw_mean]
    #
    # res = res + f1_macro_each + f1_weighted_each + rec_macro_each + rec_weighted_each + prec_macro_each \
    #       + prec_weighted_each  # + roc_auc_macro_each + roc_auc_weighted_each

    return res


def get_classification_metrics(true_classes, pred_classes, num_classes, round_decimals, model_id, model_folder,
                               with_cross_val, fold_id=None, thres=None):
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
    :param model_id: Model id (int)
    :param model_folder: The folder where the result will be saved
    :param with_cross_val: boolean (cross_validation or not)
    :param fold_id: Cross-validation fold id (int)
    :return: List of classification metrics
    """

    def compute_accuracy_per_class(is_weighted):
        """
        Compute a list of accuracies per emotion.

        :param is_weighted: whether the accuracy is weighted or not
        :return: a list of rounded accuracies
        """
        acc_list = []
        ext_name = "dom" if thres is None else "pres_t_{}".format(thres)
        ext_name += "_cv" if with_cross_val else ""

        conf_matrix = load_from_pickle('conf_matrix_{}_{}'.format(model_id, ext_name),
                                       pickle_folder="confusion_matrix", root_folder=model_folder)
        conf_matrix_i = conf_matrix[fold_id] if fold_id is not None else conf_matrix

        for emo_i in range(num_classes):
            tp = conf_matrix_i[emo_i, 0, 0]
            fp = conf_matrix_i[emo_i, 0, 1]
            fn = conf_matrix_i[emo_i, 1, 0]
            tn = conf_matrix_i[emo_i, 1, 1]
            total = tp + fp + fn + tn
            pos = tp + fp
            if is_weighted:
                # Formula from Tong, E., Zadeh, A., Jones, C., & Morency, L. P. (2017, July). Combating Human
                # Trafficking with Multimodal Deep Models. In Proceedings of the 55th Annual Meeting of the
                # Association for Computational Linguistics (Volume 1: Long Papers) (pp. 1547-1556).
                acc_emo = (tp * (total / pos) + tn) / (2 * total)
            else:
                acc_emo = (tp + tn) / (tp + fp + fn + tn)
            acc_emo_rounded = np.round(acc_emo, round_decimals)
            acc_list.append(acc_emo_rounded)
        return acc_list

    if thres is not None:
        print('\n>>> Threshold:', thres)

    # Accuracy
    bin_acc_per_class = compute_accuracy_per_class(is_weighted=False)
    wei_acc_per_class = compute_accuracy_per_class(is_weighted=True)
    multilabel_acc = np.round(accuracy_score(true_classes, pred_classes),
                              round_decimals)  # True and pred class vectors must exactly match
    bin_acc_macro = np.round(np.mean(bin_acc_per_class), round_decimals)
    wei_acc_macro = np.round(np.mean(wei_acc_per_class), round_decimals)
    print("Binary accuracy (for each):", bin_acc_per_class)
    print("Weighted accuracy (for each):", wei_acc_per_class)
    print("Multilabel accuracy:", multilabel_acc)
    print("Binary accuracy (unweighted mean):", bin_acc_macro)
    print("Weighted accuracy (unweighted mean):", wei_acc_macro)

    # F1 score
    f1_each = f1_score(true_classes, pred_classes, average=None, zero_division=1)
    f1_each_rounded = [np.round(val, round_decimals) for val in f1_each]
    f1_macro = np.round(f1_score(true_classes, pred_classes, average='macro', zero_division=1), round_decimals)
    # f1_weighted = np.round(f1_score(true_classes, pred_classes, average='weighted', zero_division=1), round_decimals)
    print("F1 score (for each):", f1_each_rounded)
    print("F1 score (unweighted mean):", f1_macro)
    # print("F1 score (weighted mean):", f1_weighted)  # biased towards the over-presence of negative classes

    # Recall
    rec_each = recall_score(true_classes, pred_classes, average=None, zero_division=1)
    rec_each_rounded = [np.round(val, round_decimals) for val in rec_each]
    rec_macro = np.round(recall_score(true_classes, pred_classes, average='macro', zero_division=1), round_decimals)
    # rec_weighted = np.round(recall_score(true_classes, pred_classes, average='weighted', zero_division=1),
    #                         round_decimals)
    print("Recall (for each):", rec_each_rounded)
    print("Recall (unweighted mean):", rec_macro)
    # print("Recall (weighted mean):", rec_weighted)  # biased towards the over-presence of negative classes

    # Precision
    prec_each = precision_score(true_classes, pred_classes, average=None, zero_division=1)
    prec_each_rounded = [np.round(val, round_decimals) for val in prec_each]
    prec_macro = np.round(precision_score(true_classes, pred_classes, average='macro', zero_division=1), round_decimals)
    # prec_weighted = np.round(precision_score(true_classes, pred_classes, average='weighted', zero_division=1),
    #                          round_decimals)
    print("Precision (for each):", prec_each_rounded)
    print("Precision (unweighted mean):", prec_macro)
    # print("Precision (weighted mean):", prec_weighted)  # biased towards the over-presence of negative classes

    # ROC AUC score
    true_classes_bin = label_binarize(true_classes, classes=list(range(num_classes)))
    pred_classes_bin = label_binarize(pred_classes, classes=list(range(num_classes)))
    roc_auc_each = roc_auc_score(true_classes_bin, pred_classes_bin, average=None)
    roc_auc_macro = np.round(roc_auc_score(true_classes_bin, pred_classes_bin, average='macro'), round_decimals)
    # roc_auc_weighted = np.round(
    #     roc_auc_score(true_classes_bin, pred_classes_bin, average='weighted', multi_class='ovr'), round_decimals)
    roc_auc_each_rounded = [np.round(val, round_decimals) for val in roc_auc_each]
    print("ROC AUC (for each):", roc_auc_each_rounded)
    print("ROC AUC (unweighted mean):", roc_auc_macro)
    # print("ROC AUC (weighted mean):", roc_auc_weighted)

    res = [multilabel_acc, bin_acc_macro, wei_acc_macro, f1_macro, rec_macro, prec_macro, roc_auc_macro]
    res = res + bin_acc_per_class + wei_acc_per_class + f1_each_rounded + rec_each_rounded + prec_each_rounded \
          + roc_auc_each_rounded

    return res


def plot_multilabel_confusion_matrix(num_classes, with_cross_val, num_folds, model_id, model_folder, label_type,
                                     display_fig, threshold_emo_pres):
    display = None

    def plot_one_confusion_matrix(conf_matrix, emo_i, axe_id, end_of_title):
        d = ConfusionMatrixDisplay(confusion_matrix=conf_matrix[emo_i, :, :], display_labels=[0, 1])
        d.plot(ax=axes[axe_id], values_format='.4g')
        d.ax_.set_title(emotions[emo_i] + end_of_title)
        # if i != len(threshold_emo_pres)-1:  ## Remove "Predicted label" xlabel from the graph
        #     display.ax_.set_xlabel('')
        if emo_i != 0:
            d.ax_.set_ylabel('')
        d.im_.colorbar.remove()

        return d

    def cross_valid_plot_matrix(conf_matrix, line_id, thres=None):
        conf_mean = np.mean(conf_matrix, axis=0)
        # conf_std = np.std(conf_matrix, axis=0)
        # conf_for_print = np.array(["{} \u00B1 {}".format(conf_mean[i, j, k], conf_std[i, j, k])
        #                            for i in range(num_classes) for j in range(2) for k in range(2)]) \
        #     .reshape(conf_mean.shape)
        end_of_title = " dominant (mean of {} folds)".format(num_folds) if thres is None \
            else " with score>{} (mean of {} folds)".format(thres, num_folds)
        for emo_i in range(num_classes):
            display = plot_one_confusion_matrix(conf_mean, emo_i, axe_id=line_id * num_classes + emo_i,
                                                end_of_title=end_of_title)
        return display

    def hold_out_plot_matrix(conf_matrix, line_id, thres=None):
        end_of_title = " dominant" if thres is None else " with score>{}".format(thres)
        for emo_i in range(num_classes):
            display = plot_one_confusion_matrix(conf_matrix, emo_i, axe_id=line_id * num_classes + emo_i,
                                                end_of_title=end_of_title)
        return display

    emotions = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    if num_classes == 7:
        emotions += ['neutral']

    if label_type == 'dominant':
        f, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))
    else:
        f, axes = plt.subplots(len(threshold_emo_pres) + 1, num_classes,
                               figsize=(5 * num_classes, 5 * (len(threshold_emo_pres) + 1)))

    axes = axes.ravel()

    ext_name = "_cv" if with_cross_val else ""

    # Plot confusion matrix for dominant emotions
    conf_matrix_dom = load_from_pickle('conf_matrix_{}_dom{}'.format(model_id, ext_name),
                                       pickle_folder="confusion_matrix", root_folder=model_folder)
    if with_cross_val:
        display = cross_valid_plot_matrix(conf_matrix_dom, line_id=0)
    else:
        display = hold_out_plot_matrix(conf_matrix_dom, line_id=0)

    # Plot confusion matrix for emotions present
    if label_type != 'dominant':
        for i, thres in enumerate(threshold_emo_pres):
            conf_matrix_pres = load_from_pickle('conf_matrix_{}_pres_t_{}{}'.format(model_id, thres, ext_name),
                                                pickle_folder="confusion_matrix", root_folder=model_folder)
            if with_cross_val:
                display = cross_valid_plot_matrix(conf_matrix_pres, line_id=i + 1, thres=thres)
            else:
                display = hold_out_plot_matrix(conf_matrix_pres, line_id=i + 1, thres=thres)

    # Save figure
    plt.savefig(os.path.join(model_folder, "confusion_matrix", "conf_matrix_{}.png".format(model_id)),
                bbox_inches='tight')

    f.colorbar(display.im_, ax=axes)
    if display_fig:
        plt.show()


def compute_multilabel_confusion_matrix(true_classes, pred_classes, with_cross_val, num_classes,
                                        model_id, model_folder, threshold_emo_pres=None):
    """
    Compute the multilabel confusion matrix for each emotion (whether the emotion is present or not).
    Pickle file generated.

    :param true_classes: List of arrays giving the emotions truly present (varying thresholds for presence score
    given by threshold_emo_pres) or the dominant emotions
    :param pred_classes: List of arrays giving the emotions present/dominant predicted by the model
    :param with_cross_val: boolean (cross_validation or not)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param model_id: Model id (int)
    :param model_folder: The folder where the results will be saved
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3.
    This parameter should be used only when dealing with emotions present.
    """

    def cross_valid_conf_matrix(thres=None, thres_id=None):
        conf_matrices = []
        num_folds = len(pred_classes)  # pred_classes is a list of num_fold elements
        task_name = "pres_t_{}".format(thres) if thres is not None else "dom"

        for fold_id in range(num_folds):
            tc = true_classes[thres_id] if thres_id is not None else true_classes
            pc = pred_classes[fold_id][thres_id] if thres_id is not None else pred_classes[fold_id]
            conf_matrix = multilabel_confusion_matrix(tc, pc, labels=list(range(num_classes)))
            conf_matrices += [conf_matrix]
        save_with_pickle(conf_matrices, 'conf_matrix_{}_{}_cv'.format(model_id, task_name),
                         pickle_folder="confusion_matrix", root_folder=model_folder)

    def hold_out_conf_matrix(thres=None, thres_id=None):
        task_name = "pres_t_{}".format(thres) if thres is not None else "dom"
        tc = true_classes[thres_id] if thres_id is not None else true_classes
        pc = pred_classes[thres_id] if thres_id is not None else pred_classes
        conf_matrix = multilabel_confusion_matrix(tc, pc, labels=list(range(num_classes)))
        save_with_pickle(conf_matrix, 'conf_matrix_{}_{}'.format(model_id, task_name),
                         pickle_folder="confusion_matrix", root_folder=model_folder)

    # emotions present
    if threshold_emo_pres is not None:
        for thres_id, thres in enumerate(threshold_emo_pres):
            if with_cross_val:
                cross_valid_conf_matrix(thres, thres_id)
            else:
                hold_out_conf_matrix(thres, thres_id)

    # emotions dominant
    else:
        if with_cross_val:
            cross_valid_conf_matrix()
        else:
            hold_out_conf_matrix()


def get_and_print_all_metrics(label_type, true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom,
                              pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom, with_cross_val,
                              threshold_emo_pres, num_classes, predict_neutral_class, round_decimals,
                              model_id, model_folder):
    """
    Compute metrics for four tasks: estimation of presence score (regression), classification of presence score among
    [0, 1, 2, 3], binary classification for each emotion, and prediction of dominant classes (multilabel classification)

    :param label_type: Type of labels used for model training and evaluation.
    :param true_scores_all: array of shape (test_size, num_classes) giving the true presence score of the emotions
    :param true_scores_coa: array of shape (test_size, num_classes) giving the true presence score among [0, 1, 2, 3]
    :param true_classes_pres: list of binary arrays of shape (test_size, num_classes) giving the true presence of an
    emotion (> threshold defined by threshold_emo_pres)
    :param true_classes_dom: binary array of shape (test_size, num_classes) giving the truly dominant emotions
    :param pred_raw: array of shape (test_size, num_classes) predicting the presence score of the emotions
    :param pred_scores_coa: array of shape (test_size, num_classes) predicting the presence score among [0, 1, 2, 3]
    :param pred_classes_pres: list of binary arrays of shape (test_size, num_classes) giving the predicted presence of
    an emotion (> threshold defined by threshold_emo_pres)
    :param pred_classes_dom: binary array of shape (test_size, num_classes) giving the predicted dominant emotions
    :param with_cross_val: boolean (cross_validation or not)
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param predict_neutral_class: Whether we predict the neutral class
    :param round_decimals: Number of decimals to be rounded for metrics
    :param model_id: Model id (int)
    :param model_folder: The folder where the result will be saved
    :return: metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant
    """

    print("\n\n")
    print("In the following, all the metrics displayed for each emotion are in this order: happiness, sadness, "
          "anger, surprise, disgust, fear", end="")
    print(", neutral.\n") if predict_neutral_class else print(".\n")

    # Some metrics do not make sense for the targeted task (label_type): we do not print them
    if label_type == 'all':
        compute_reg, compute_coa, compute_pres = True, True, True
    elif 'coarse' in label_type:
        compute_reg, compute_coa, compute_pres = False, True, True
    elif label_type == 'present':
        compute_reg, compute_coa, compute_pres = False, False, True
    else:  # dominant
        compute_reg, compute_coa, compute_pres = False, False, False

    def compute_metrics_one_model(pred_raw_i, pred_scores_coa_i, pred_classes_pres_i, pred_classes_dom_i, fold_id=None):
        # Regression metrics
        if compute_reg:
            print("\n------- Presence score estimation (regression) --------\n")
            metrics_regression = get_regression_metrics(true_scores_all, pred_raw_i, round_decimals)
        else:
            metrics_regression = None
            # metrics_regression = ['-' for _ in range(21)]  # 21 regression metric values replaced by "-"

        # Classification metrics of presence scores
        if compute_coa:
            print("\n------- Presence score classification [0,1,2,3] -------\n")
            metrics_score_coa = get_classification_metrics_score_coa(true_scores_coa, pred_scores_coa_i, num_classes,
                                                                     round_decimals, prob=('prob' in label_type))
        else:
            metrics_score_coa = None
            # metrics_score_coa = ['-' for _ in range(43)]

        # Classification metrics of emotion present
        if compute_pres:
            print("\n------- Detecting the presence of emotions ------------")
            metrics_presence = [get_classification_metrics(true_classes_pres[j], pred_classes_pres_i[j], num_classes,
                                                           round_decimals, model_id, model_folder, with_cross_val,
                                                           fold_id, thres)
                                for j, thres in enumerate(threshold_emo_pres)]
        else:
            metrics_presence = None

        # Classification metrics of dominant emotions
        print("\n------- Predicting dominant emotions ------------------\n")
        metrics_dominant = get_classification_metrics(true_classes_dom, pred_classes_dom_i, num_classes, round_decimals,
                                                      model_id, model_folder, with_cross_val, fold_id)

        return metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant

    if with_cross_val:
        met_reg, met_coa, met_pres, met_dom = [], [], [], []
        for fold_id, (pr_i, pc_i, pp_i, pd_i) in enumerate(zip(pred_raw, pred_scores_coa,
                                                               pred_classes_pres, pred_classes_dom)):
            mr, mc, mp, md = compute_metrics_one_model(pr_i, pc_i, pp_i, pd_i, fold_id)
            met_reg += [mr]
            met_coa += [mc]
            met_pres += [mp]
            met_dom += [md]
    else:
        met_reg, met_coa, met_pres, met_dom = compute_metrics_one_model(pred_raw, pred_scores_coa,
                                                                        pred_classes_pres, pred_classes_dom)
    return met_reg, met_coa, met_pres, met_dom


def compute_and_save_model_metrics(test_list, with_cross_val, model_eval_output,
                                   pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom,
                                   param_task, param_training, param_log, model_name, model_id, all_scores):
    """
    Compute and save model metrics, based on loss function value and prediction scores.

    :param test_list: [x_test, y_test, seg_test]
    :param with_cross_val: boolean (cross_validation or not)
    :param model_eval_output: List of [loss function, validation metric] obtained by the model
    :param pred_raw: array of shape (test_size, num_classes) predicting the presence score of the emotions
    :param pred_scores_coa: array of shape (test_size, num_classes) predicting the presence score among [0, 1, 2, 3]
    :param pred_classes_pres: list of binary arrays of shape (test_size, num_classes) giving the predicted presence of
    an emotion (> threshold defined by threshold_emo_pres)
    :param pred_classes_dom: binary array of shape (test_size, num_classes) giving the predicted dominant emotions
    :param param_log: dictionary of 'logistics' hyperparameters (cf. param_logistics.json)
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param param_training: dictionary of 'training' hyperparameters (cf. param_training_task.json)
    :param model_name: Name of the model currently tested
    :param model_id: Model id (int)
    :param all_scores: list of available presence scores
    """

    predict_neutral_class = param_task['predict_neutral_class']
    threshold_emo_pres = [param_task['threshold_emo_present']]
    label_type = param_task['label_type']
    round_decimals = param_log['round_decimals']
    plot_confusion_matrix = param_log['plot_confusion_matrix']
    display_fig = param_log['display_fig']
    loss_function = param_training['loss_function']
    val_metrics_str = param_training['val_metrics']
    num_folds = param_training['cross_val_folds']

    model_folder = os.path.join('results', model_name)

    # Extract x, y and seg_ids for test set
    y_test = test_list[1]  # each element of shape (1, 6 or 7)
    num_classes = y_test[0].shape[1]

    # True labels
    true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom = \
        compute_true_labels(y_test, "test", all_scores, predict_neutral_class, threshold_emo_pres, num_classes,
                            label_type, return_all_tasks=True)

    # Confusion matrix
    if label_type != 'dominant':
        compute_multilabel_confusion_matrix(true_classes_pres, pred_classes_pres, with_cross_val, num_classes,
                                            model_id, model_folder, threshold_emo_pres)
    compute_multilabel_confusion_matrix(true_classes_dom, pred_classes_dom, with_cross_val, num_classes,
                                        model_id, model_folder)
    if plot_confusion_matrix:
        plot_multilabel_confusion_matrix(num_classes, with_cross_val, num_folds, model_id, model_folder, label_type,
                                         display_fig, threshold_emo_pres)

    # Compute all metrics
    metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant = \
        get_and_print_all_metrics(label_type, true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom,
                                  pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom, with_cross_val,
                                  threshold_emo_pres, num_classes, predict_neutral_class, round_decimals,
                                  model_id, model_folder)

    # Save metrics in csv file
    save_results_in_csv_file(model_name, model_id, loss_function, val_metrics_str, model_eval_output,
                             metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant,
                             with_cross_val, predict_neutral_class, threshold_emo_pres)
