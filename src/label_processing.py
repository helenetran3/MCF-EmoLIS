import numpy as np
import re
from src.pickle_functions import *

from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer


def trunc(val, decimals):
    stepper = 10 ** decimals
    return np.trunc((val * stepper)) / stepper


def create_label_multitask(label_type):
    if len(label_type) != 2:
        raise ValueError("The list format for label_type is reserved for multitask and must contain exactly 2 elements."
                         " Please change in config/param_training_task.json.")
    granularity_deg = {'all': 1, 'coarse': 2, 'present': 3, 'dominant': 4}
    label_0, label_1 = label_type[0], label_type[1]
    label_0 = 'coarse' if 'coarse' in label_0 else label_0
    label_1 = 'coarse' if 'coarse' in label_1 else label_1
    if label_0 == label_1:
        raise ValueError("The elements of the list 'label_type' should not be the same. "
                         "Please change in config/param_training_task.json.")
    if granularity_deg[label_0] > granularity_deg[label_1]:
        y_label = {'coarse_task': label_type[0], 'fine_task': label_type[1]}
    else:
        y_label = {'coarse_task': label_type[1], 'fine_task': label_type[0]}
    return y_label


def shorten_label(label):
    if 'coarse' in label:
        m = re.search(r'(?<=_)\w+', label)
        return 'coa' + m.group(0) if m is not None else 'coa'
    elif label == 'present':
        return 'pres'
    else:
        return label[:3]


def create_label_filename(label_type):
    if type(label_type) == dict:
        return shorten_label(label_type['fine_task']) + '&' + shorten_label(label_type['coarse_task'])
    else:
        return shorten_label(label_type)


def from_one_hot_to_plain(scores_array, num_classes):
    # Prediction given in one_hot, go back to plain scores for model metric calculation
    # Check if the input score array is in one hot format
    if scores_array.shape[-1] != 4 or scores_array.shape[-2] != num_classes:
        raise ValueError("The output of the model is not in shape (num_samples, num_classes, num_scores=4)"
                         " as the label type requires one-hot encoding. Please check your model output.")

    all_samples_scores = []
    for sample_i in range(scores_array.shape[0]):
        sample_i_scores = []
        for class_j in range(num_classes):
            max_ind_ij = np.argmax(scores_array[sample_i, class_j])  # array (4,) for a given sample and class
            # sample_i_scores += [max_ind_ij]
            sample_i_scores += [[int(k == max_ind_ij) for k in range(4)]]
        sample_i_scores_plain = np.argmax(np.array(sample_i_scores), axis=1)
        all_samples_scores += [list(sample_i_scores_plain)]
    out_scores = np.array(all_samples_scores)
    # out_scores = np.reshape(np.array(all_samples_scores), (-1, num_classes))
    return out_scores


def get_available_presence_scores(true_scores_all):
    le = LabelEncoder()
    le.fit(true_scores_all.flatten())
    return le.classes_


def get_available_presence_scores_arr(y_list):
    """
    Get the list of all unique values of presence scores, given a list of y arrays of shape (1, 6)

    :param y_list: a list of y arrays of shape (1, 6)
    :return: a list of unique values of presence scores
    """
    all_y = np.reshape(np.array(y_list), (-1, 6))
    all_y_uniq = np.unique(all_y)
    return list(all_y_uniq)


def get_coa_presence_score_for_training(y_list, num_classes, one_hot, prob):
    scores = [0, 1, 2, 3]
    scores_array = np.concatenate(y_list, axis=0).flatten()

    # The following line returns the class closest to the input value
    out_scores = [min(list(scores), key=lambda x: abs(x - scores_array[i])) for i in range(scores_array.shape[0])]

    if one_hot:
        # One-hot encode the true labels (that will be used for model training)
        lb = LabelBinarizer()
        _ = lb.fit(scores)
        out_scores = lb.transform(out_scores)
        out_scores = np.reshape(np.array(out_scores), (-1, num_classes, len(scores)))

    if prob:
        out_scores = [trunc(sc_i / 3, decimals=2) for sc_i in out_scores]
        out_scores = np.reshape(np.array(out_scores), (-1, num_classes))

    if not one_hot and not prob:
        out_scores = np.reshape(np.array(out_scores), (-1, num_classes))

    return out_scores


def get_presence_score_for_metrics(scores_array, all_scores, num_classes, coarse):
    """
    Get the presence score from a finer grained presence score
    (raw predictions -> 12 or 4 presence scores, or 12 presence scores -> 4 presence scores)

    :param scores_array: input array of shape (test_size, num_classes) giving the presence score of the 6 emotions
    :param all_scores: list of available presence scores [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    :param num_classes: number of emotion classes (7 with neutral class, else 6)
    :param coarse: if True, the resulting presence scores in [0, 1, 2, 3], else all_scores
    :return: array of shape (test_size, num_classes) [or (test_size, num_classes, num_scores) if one_hot] giving the
    presence score of all the 6 emotions
    """

    scores_val = all_scores if not coarse else [0, 1, 2, 3]
    scores_array = scores_array.flatten()

    # The following line returns the class closest to the input value
    out_scores = [min(list(scores_val), key=lambda x: abs(x - scores_array[i])) for i in range(scores_array.shape[0])]
    out_scores = np.reshape(np.array(out_scores), (-1, num_classes))

    return out_scores


def get_class_from_presence_score(score_array, with_neutral_class, thres_score_pres, thres_multilabel,
                                  num_classes, is_dominant, is_true_labels, label_type):
    """
    Get the dominant emotion class(es) for each test segment based on the presence scores of emotions.
    If is_dominant, return a binary array, else return a list of len(thres_score_pres) binary arrays.
    Column indexes: 0:happy, 1:sad, 2:anger, 3:surprise, 4:disgust, 5:fear, (6:neutral)

    :param score_array: array of shape (test_size, num_classes) containing the presence score of 6 emotions (+ neutral)
    :param with_neutral_class: whether we add a neutral class or not
    :param thres_score_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param thres_multilabel: list of num_classes elements which give the emotion-wise thresholds above which the
     emotion class is considered as positive, either the ones giving the highest F1 score from validation set
     (thres_ml_from_best_metric=1) or 0.5 (thres_ml_from_best_metric=0). Set to None only when model training has not
     been started yet (for true_label calculation)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param is_dominant: Whether we calculate dominant classes or not (else we calculate the presence of classes)
    :param is_true_labels: Whether the score_array represent true labels or not (else it is predicted)
    :param label_type: Type of labels used for model training and evaluation. Possible values: all (for all presence
                         scores provided by the dataset), coarse (presence score for each emotion among [0, 1, 2, 3]),
                         coarse_prob (presence score for each emotion among [0, 1/3, 2/3, 1]),
                         present (multi-label classification), dominant (multi-label classification)
    :return: Binary array of shape (test_size,7) if we add neutral class, else array of shape (test_size, 6).
    """

    def get_list_emotions_for_each_seg(thres_seg=None):
        """
        Get the list of all emotion ids for each segment.
        For example, if sad is index 1 and i-th segment only shows sadness, then the i-th element of the list is [1].
        If j-th segment shows no emotion, then j-th element of the list is [6] if we predict neutral class else [].
        :param thres_seg: Threshold above which the emotion is considered as being present.
        :return: list of lists of emotion ids. The number of elements of this list is the number of segments.
        """
        list_emo_segs = []
        test_size = score_array.shape[0]

        for seg_id in range(test_size):  # we check every segment

            score_seg = score_array[seg_id, :]

            max_score = max(score_seg)
            if max_score == 0:  # the current segment shows neutral (no emotions)
                list_emo_segs.append([6]) if with_neutral_class else list_emo_segs.append([])
            elif is_dominant:
                # the following takes the index(es) of the maximum value in the presence score vector of emotions
                list_emo_segs.append([i for i, val in enumerate(score_seg) if val == max_score])
            else:
                # For emotion present
                tmp = [i for i, val in enumerate(score_seg) if val > thres_seg]
                tmp = [6] if len(tmp) == 0 and with_neutral_class else tmp  # just for safety
                list_emo_segs.append(tmp)

        return list_emo_segs

    # Transform list of indexes to one-hot vector
    mlb = MultiLabelBinarizer(classes=list(range(num_classes)))

    # In this case, we work with true or estimated presence scores
    if is_true_labels or (not is_true_labels and (label_type == 'all' or 'coarse' in label_type)):
        if is_dominant:
            # Infer dominant emotions by taking the highest presence score
            list_emotions = get_list_emotions_for_each_seg()
            bin_array_emotions = mlb.fit_transform(list_emotions)
            return bin_array_emotions
        else:
            # Infer present emotions from true or predicted presence scores, using threshold set by the user
            # Happens in two cases: when computing true labels, or when the predicted labels are presence scores
            # ('all' or 'coarse')
            list_emotions_all_thresholds = []
            for thres in thres_score_pres:  # Loop over all threshold values
                list_emotions_thres = get_list_emotions_for_each_seg(thres_seg=thres)
                list_emotions_all_thresholds.append(list_emotions_thres)

            list_bin_array_emotions = [mlb.fit_transform(l) for l in list_emotions_all_thresholds]
            return list_bin_array_emotions

    # Infer dominant or present emotions from the predicted model output, where the model perform
    # multi-label classification (so values are between 0 and 1), using thres_multilabel
    else:
        bin_array = np.empty(score_array.shape)
        for cl_i in range(num_classes):
            score_cl_i = score_array[:, cl_i]
            score_cl_bin_i = np.empty(score_cl_i.shape)
            score_cl_bin_i[score_cl_i > thres_multilabel[cl_i]] = 1
            score_cl_bin_i[score_cl_i <= thres_multilabel[cl_i]] = 0
            bin_array[:, cl_i] = score_cl_bin_i
        assert ((bin_array == 0) | (bin_array == 1)).all()  # Check if bin_array is binary
        return bin_array if is_dominant else [bin_array]  # for the moment we create a list


def compute_true_labels(y_list, root_name, all_scores, predict_neutral_class, thres_score_pres, param_task, num_classes,
                        label_type, for_test=False):
    """
    Compute the true labels from a list of 2D arrays of shape (1, num_classes) (y_list)
    The results are all 3D arrays, true_classes_pres is a list of arrays of this type)
    true_scores_all: Presence scores provided by the dataset. Shape: (number of y elements, 6)
    true_scores_coa: Closest true score among [0, 1, 2, 3]. Shape: (number of y elements, 6)
    true_classes_pres: Emotions truly present (varying thresholds for presence score given by thres_score_pres).
    Shape: (number of y elements, num_classes)
    true_classes_dom: Emotions truly dominant (highest presence score). Shape: (number of y elements, num_classes)

    :param y_list: List of arrays of shape (1, num_classes): contains original scores (for_test False) or the labels
                     computed from the targeted task (for_test True)
    :param root_name: Name to append to pickle file (ex: 'train', 'valid', 'test')
    :param all_scores: list of available presence scores
    :param predict_neutral_class: Whether we predict the neutral class
    :param thres_score_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param num_classes: number of emotion classes (7 with neutral class, else 6)
    :param label_type: Type of labels used for model training and evaluation. Possible values: all (for all presence
                         scores provided by the dataset), coarse (presence score for each emotion among [0, 1, 2, 3]),
                         coarse_prob (presence score for each emotion among [0, 1/3, 2/3, 1]),
                         present (multi-label classification), dominant (multi-label classification)
    :param for_test: Whether we return the true vectors of all applicable tasks based on label type (True) or return
                       only the vector corresponding to the label type (False)
    :return: true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom depending on return_all_tasks
    """

    # In the following lines, some labels do not make sense for the targeted task (label_type): we set them as None
    true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom = None, None, None, None
    # Used to compute derived and applicable tasks
    compute_coa, compute_pres, compute_dom = False, False, False
    y_array = np.reshape(np.array(y_list), (-1, 6))

    # In the case of test (for metric calculation), assign the 2D y_array to its corresponding targeted task
    if for_test:
        if label_type == 'all':
            true_scores_all = y_array
            compute_coa, compute_pres, compute_dom = True, True, True
        elif 'coarse' in label_type:
            true_scores_coa = y_array
            compute_pres, compute_dom = True, True
        elif label_type == 'present':
            true_classes_pres = [y_array]
        else:  # dominant
            true_classes_dom = y_array
    # Else, assign the 2D y_array to true_scores_all (used at the beginning to infer the targeted task)
    else:
        true_scores_all = y_array

    # Compute true presence scores: arrays of shape (4654, 6)
    # Possible values: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    # Coarse-grained values: [0, 1, 2, 3] if not prob else [0, 1/3, 2/3, 1]
    if 'coarse' in label_type and not for_test:
        # For training
        true_scores_coa = get_coa_presence_score_for_training(y_array, num_classes,
                                                              prob=('prob' in label_type),
                                                              one_hot=('oh' in label_type))
    if compute_coa and for_test:
        # For metric calculation
        true_scores_coa = get_presence_score_for_metrics(y_array, all_scores, num_classes, coarse=True)

    # Compute true present classes: binary arrays of shape (num_test, 6 or 7)
    if (label_type == 'present' and not for_test) or (compute_pres and for_test):
        true_classes_pres = []
        for id, thres in enumerate(thres_score_pres):
            true_classes_pres_i = get_class_from_presence_score(y_array, predict_neutral_class, [thres],
                                                                None, num_classes, is_dominant=False,
                                                                is_true_labels=True, label_type=label_type)
            true_classes_pres += true_classes_pres_i

    # Compute true dominant classes: binary arrays of shape (num_test, 6 or 7)
    if (label_type == 'dominant' and not for_test) or (compute_dom and for_test):
        true_classes_dom = get_class_from_presence_score(y_array, predict_neutral_class, None, None,
                                                         num_classes, is_dominant=True, is_true_labels=True,
                                                         label_type=label_type)

    if for_test:
        return true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom
    else:
        if label_type == 'all':
            return true_scores_all
        elif 'coarse' in label_type:
            return true_scores_coa
        elif label_type == 'present':
            return true_classes_pres
        elif label_type == 'dominant':
            return true_classes_dom
        else:
            return ValueError("Value label_type invalid.")


def compute_pred_labels(pred_raw, predict_neutral_class, thres_score_pres, thres_multilabel, save_pred, model_id,
                        num_classes, model_folder, label_type):
    """
    Compute the prediction labels (all arrays of shape (test_size, 6), pred_classes_pres is a list of arrays of this type)
    pred_scores_coa: Closest predicted score among [0, 1, 2, 3]
    pred_classes_pres: Emotions present predicted by the model (varying thresholds for presence score given by thres_score_pres)
    pred_classes_dom: Dominant emotions predicted by the model (highest presence score)
    :param pred_raw: array of shape (test_size, num_classes) predicting the presence score of the emotions (in [0,3])
    :param predict_neutral_class: Whether we predict the neutral class
    :param thres_score_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param thres_multilabel: list of num_classes elements which give the emotion-wise thresholds above which the
     emotion class is considered as positive, either the ones giving the highest F1 score from validation set
     (thres_ml_from_best_metric=1) or 0.5 (thres_ml_from_best_metric=0). Set to None only when model training has not
     been started yet (for true_label calculation)
    :param save_pred: Whether we save predictions with pickle
    :param model_id: Model id (int)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param model_folder: The folder where the results will be saved
    :param label_type: Type of labels used for model training and evaluation.
    :return: pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom
    """

    # In the following lines, some labels do not make sense for the targeted task (label_type): we set them as None
    pred_classes_pres, pred_classes_dom = None, None
    if label_type == 'all' or 'coarse' in label_type:
        compute_pres, compute_dom = True, True
    elif label_type == 'present':
        compute_pres, compute_dom = True, False
    else:  # dominant
        compute_pres, compute_dom = False, True

    # Save raw predictions
    if save_pred:
        save_with_pickle(pred_raw, "pred_raw_{}".format(model_id), pickle_folder="predictions",
                         root_folder=model_folder)

    print("\n\nTargeted task:", label_type)

    # Get present emotion classes predicted by the model, based on thresholds
    if compute_pres:
        pred_classes_pres = get_class_from_presence_score(pred_raw, predict_neutral_class, thres_score_pres,
                                                          thres_multilabel, num_classes, is_dominant=False,
                                                          is_true_labels=False, label_type=label_type)
        if label_type == 'all' or 'coarse' in label_type:
            pres_calc_str = "using the thres_score_pres={} set by the user".format(thres_score_pres)
        else:
            pres_calc_str = "using the following class-wise thresholds: " + str(thres_multilabel)
        print("pred_classes_pres computed: Presence of emotion classes {}.".format(pres_calc_str))
        if save_pred:
            save_with_pickle(pred_classes_pres, "pred_classes_pres_{}".format(model_id), pickle_folder="predictions",
                             root_folder=model_folder)

    # Get dominant emotion classes predicted by the model
    if compute_dom:
        pred_classes_dom = get_class_from_presence_score(pred_raw, predict_neutral_class, None, thres_multilabel,
                                                         num_classes, is_dominant=True, is_true_labels=False,
                                                         label_type=label_type)
        if label_type == 'all' or 'coarse' in label_type:
            dom_calc_str = "as the highest score from prediction"
        else:
            dom_calc_str = "using the following class-wise thresholds: " + str(thres_multilabel)
        print("pred_classes_dom computed: Dominance of the emotion classes {}.".format(dom_calc_str))
        if save_pred:
            save_with_pickle(pred_classes_dom, "pred_classes_dom_{}".format(model_id), pickle_folder="predictions",
                             root_folder=model_folder)

    return pred_classes_pres, pred_classes_dom


def update_labels(train_list, valid_list, test_list, no_split_list, param_task, label_type, all_scores, data_ext_name,
                  root_data_folder):
    """
    Update labels for each fold so that it matches with the targeted task given by label_type:
    - all: original presence scores provided by the dataset
    - coarse: presence score for each emotion among [0, 1, 2, 3]
    - coarse_prob: presence score for each emotion among [0, 1/3, 2/3, 1]
    - present: one-hot encoded labels for multi-label classification. 7 classes if predict_neutral_class else 6.
    - dominant: one-hot encoded labels for one-label classification. 7 classes if predict_neutral_class else 6.
    It also removes samples assigned to neutral when predict_neutral_class is False + return the indexes of the neutral
    samples in test set.

    :param train_list: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
    :param valid_list: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
    :param test_list: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
    :param no_split_list: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param all_scores: list of available presence scores
    :return: new_train_list, new_valid_list, new_test_list, with the same structure as the input, + neutral_test_ind
    """

    def change_y(train_l, valid_l, test_l, no_split_l):

        y1, y2, y3 = train_l[1], valid_l[1], test_l[1]
        y4 = no_split_l[1] if no_split_l is not None else None

        def change_type_and_shape_for_training(y_array):
            """
            Create a list of arrays of shape (1, num_classes)
            :param y_array: array of shape (num_samples, num_classes)
            :return: a list of arrays of shape (1, num_classes)
            """
            num_output = y_array.shape[-1]
            y_list = list(y_array)
            y_list_res = [y.reshape((-1, num_output)) for y in y_list]
            return y_list_res

        def change_y_single_task(label_t):
            new_y1 = compute_true_labels(y1, "train", all_scores, predict_neutral_class, thres_score_pres,
                                         param_task, num_classes, label_t)
            new_y2 = compute_true_labels(y2, "valid", all_scores, predict_neutral_class, thres_score_pres,
                                         param_task, num_classes, label_t)
            new_y3 = compute_true_labels(y3, "test", all_scores, predict_neutral_class, thres_score_pres,
                                         param_task, num_classes, label_t)
            new_y4 = compute_true_labels(y4, "no_split", all_scores, predict_neutral_class, thres_score_pres,
                                         param_task, num_classes, label_t) if y4 is not None else None

            if label_t == 'present':  # TODO We only take the first threshold for the moment
                new_y1 = new_y1[0]
                new_y2 = new_y2[0]
                new_y3 = new_y3[0]
                new_y4 = new_y4[0] if new_y4 is not None else None

            # Adapt to model training and evaluation
            new_y1 = change_type_and_shape_for_training(new_y1)
            new_y2 = change_type_and_shape_for_training(new_y2)
            new_y3 = change_type_and_shape_for_training(new_y3)
            new_y4 = change_type_and_shape_for_training(new_y4) if new_y4 is not None else None

            return new_y1, new_y2, new_y3, new_y4

        # Multitask case
        if type(label_type) == dict:
            coa_y1, coa_y2, coa_y3, coa_y4 = change_y_single_task(label_type['coarse_task'])
            fin_y1, fin_y2, fin_y3, fin_y4 = change_y_single_task(label_type['fine_task'])
            new_y_train = {'coarse_task': coa_y1, 'fine_task': fin_y1}
            new_y_valid = {'coarse_task': coa_y2, 'fine_task': fin_y2}
            new_y_test = {'coarse_task': coa_y3, 'fine_task': fin_y3}
            new_y_no_split = {'coarse_task': coa_y4, 'fine_task': fin_y4} if no_split_list is not None else None

        else:
            new_y_train, new_y_valid, new_y_test, new_y_no_split = change_y_single_task(label_type)

        new_train_l = [train_list[0], new_y_train, train_list[2]]
        new_valid_l = [valid_list[0], new_y_valid, valid_list[2]]
        new_test_l = [test_list[0], new_y_test, test_list[2]]
        new_no_split_l = [no_split_list[0], new_y_no_split, no_split_list[2]] if no_split_list is not None else None

        return new_train_l, new_valid_l, new_test_l, new_no_split_l  # new_no_split_l is None if no_split_list is None

    def remove_neutral_class(train_l, valid_l, test_l, no_split_l):
        """
        Remove samples assigned to neutral class + return the list of their indexes for test split

        :param train_l: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
        :param valid_l: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
        :param test_l: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
        :param no_split_l: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
        :return: new_train_l, new_valid_l, new_test_l, new_no_split_l
        """

        def remove_neutral_class_split(split_list):
            y = split_list[1]
            if type(y) == dict:
                non_neutral_ind = []
                for task_i in ['coarse_task', 'fine_task']:
                    y_arr = np.vstack(y[task_i])
                    num_emo = np.sum(y_arr, axis=1)
                    non_neutral_ind += [num_emo != 0]  # Boolean array of shape (split_size,)
                non_neutral_ind = np.all(non_neutral_ind, axis=0)
                num_neutral = y_arr.shape[0] - np.sum(non_neutral_ind)

            else:
                y_arr = np.vstack(y)
                num_emo = np.sum(y_arr, axis=1)
                non_neutral_ind = (num_emo != 0)  # Boolean array of shape (split_size,)
                num_neutral = np.sum((num_emo == 0))

            # Take only samples where at least one emotion class is assigned
            X_list = [elt for (elt, ind) in zip(split_list[0], non_neutral_ind) if ind]
            if type(y) == dict:
                y_list = {}
                for task_i in ['coarse_task', 'fine_task']:
                    y_list[task_i] = [elt for (elt, ind) in zip(split_list[1][task_i], non_neutral_ind) if ind]
            else:
                y_list = [elt for (elt, ind) in zip(split_list[1], non_neutral_ind) if ind]
            seg_list = [elt for (elt, ind) in zip(split_list[2], non_neutral_ind) if ind]

            return [X_list, y_list, seg_list], num_neutral

        new_no_split_l = None
        new_train_l, num_neutral_train = remove_neutral_class_split(train_l)
        new_valid_l, num_neutral_valid = remove_neutral_class_split(valid_l)
        new_test_l, num_neutral_test = remove_neutral_class_split(test_l)
        if no_split_l is not None:
            new_no_split_l, num_neutral_no_split = remove_neutral_class_split(no_split_l)
            num_neutral_samples = num_neutral_train + num_neutral_valid + num_neutral_test + num_neutral_no_split
            num_neutral_no_split_str = ', no_split: {}).'.format(num_neutral_no_split)
        else:
            num_neutral_samples = num_neutral_train + num_neutral_valid + num_neutral_test
            num_neutral_no_split_str = ').'

        print(f'\nRemoved {num_neutral_samples} samples with no emotions (train: {num_neutral_train}, '
              f'valid: {num_neutral_valid}, test: {num_neutral_test}' + num_neutral_no_split_str)
        num_train, num_valid, num_test = len(new_train_l[0]), len(new_valid_l[0]), len(new_test_l[0])
        num_no_split = len(new_no_split_l[0]) if no_split_l is not None else 0
        total_data = num_train + num_valid + num_test + num_no_split
        print("----------------------------------------------------")
        print("Total number samples after removing neutral class: {}".format(total_data))
        print("----------------------------------------------------")
        print("Number of training datapoints: {} ({:.2f}%)".format(num_train, 100 * (num_train / total_data)))
        print("Number of validation datapoints: {} ({:.2f}%)".format(num_valid, 100 * (num_valid / total_data)))
        print("Number of test datapoints: {} ({:.2f}%)".format(num_test, 100 * (num_test / total_data)))
        if no_split_l is not None:  # cross-validation
            print("Number of datapoints that do not belong to any split: {} ({:.2f}%)\n"
                  .format(num_no_split, 100 * (num_no_split / total_data)))

        return new_train_l, new_valid_l, new_test_l, new_no_split_l

    predict_neutral_class = param_task['predict_neutral_class']
    thres_score_pres = [param_task['thres_score_pres']]
    num_classes = 7 if predict_neutral_class else 6

    # Update y
    new_train_list, new_valid_list, new_test_list, new_no_split_list = change_y(train_list, valid_list, test_list,
                                                                                no_split_list)

    # When predict_neutral_class is False (no class created for Neutral),
    # samples with neutral class belong to any of available classes
    if not predict_neutral_class:
        new_train_list, new_valid_list, new_test_list, new_no_split_list = remove_neutral_class(new_train_list,
                                                                                                new_valid_list,
                                                                                                new_test_list,
                                                                                                new_no_split_list)

    # Saving
    label_type_str = create_label_filename(label_type)
    ho_filename = "ho_true_" + label_type_str + "_" + data_ext_name
    ho_filename += "_pred-neu" if predict_neutral_class else "_rm-neu"
    save_with_pickle(new_train_list, "train_" + ho_filename, "processed_folds", root_data_folder)
    save_with_pickle(new_valid_list, "valid_" + ho_filename, "processed_folds", root_data_folder)
    save_with_pickle(new_test_list, "test_" + ho_filename, "processed_folds", root_data_folder)
    save_with_pickle(new_no_split_list, "no_split_" + ho_filename, "processed_folds", root_data_folder)

    return new_train_list, new_valid_list, new_test_list, new_no_split_list


