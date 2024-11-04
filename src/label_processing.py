import numpy as np

from src.pickle_functions import *

from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer


def trunc(val, decimals):
    stepper = 10 ** decimals
    return np.trunc((val * stepper)) / stepper


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


def get_available_presence_scores_arr(train_list, valid_list, test_list):
    """
    Get the list of all unique values of presence scores, based on three lists of [x, y, seg] (training, validation, test)

    :param train_list: a list [x_train, y_train, seg_train]
    :param valid_list: a list [x_valid, y_valid, seg_valid]
    :param test_list: a list [x_test, y_test, seg_test]
    :return: a list of unique values of presence scores
    """
    train_y = np.reshape(np.array(train_list[1]), (-1, 6))
    valid_y = np.reshape(np.array(valid_list[1]), (-1, 6))
    test_y = np.reshape(np.array(test_list[1]), (-1, 6))
    all_y = np.concatenate([train_y, valid_y, test_y], axis=0)

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
                        label_type, return_all_tasks=False):
    """
    Compute the true labels from a list of 2D arrays of shape (1, num_classes) (y_list)
    The results are all 3D arrays, true_classes_pres is a list of arrays of this type)
    true_scores_all: Presence scores provided by the dataset. Shape: (number of y elements, 6)
    true_scores_coa: Closest true score among [0, 1, 2, 3]. Shape: (number of y elements, 6)
    true_classes_pres: Emotions truly present (varying thresholds for presence score given by thres_score_pres).
    Shape: (number of y elements, num_classes)
    true_classes_dom: Emotions truly dominant (highest presence score). Shape: (number of y elements, num_classes)

    :param y_list: List of arrays of shape (1, num_classes)
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
    :param return_all_tasks: Whether we return the true vectors of the four tasks (True) or return the vector
                               corresponding to the label type (False)
    :return: true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom depending on return_all_tasks
    """

    true_scores_coa, true_classes_pres, true_classes_dom = None, None, None

    # Filenames
    thres_nan = param_task['thres_nan']
    length_seq = param_task['set_length']
    length_str = "_length_" + str(length_seq) if type(length_seq) == int else ""
    ext_name = "_tnan_" + str(thres_nan) + length_str
    ext_name += "_pred-neu" if predict_neutral_class else "_rm-neu"
    ext_name_coa_train = '_prob' if ('prob' in label_type) else '_oh' if ('oh' in label_type) else ''
    filename_all = root_name + "_true_scores_all" + ext_name
    filename_coa_train = root_name + "_true_scores_coa" + ext_name_coa_train + '_train' + ext_name
    filename_coa_met = root_name + "_true_scores_coa" + '_met' + ext_name
    filename_pres = [root_name + "_true_classes_pres_t_" + str(thres) + ext_name for thres in thres_score_pres]
    filename_dom = root_name + "_true_classes_dom" + ext_name

    # Create array of original scores: 2D array from list of 2D true presence score arrays of shape (1, 6)
    if not pickle_file_exists(filename_all, 'processed_folds', 'cmu_mosei'):
        true_scores_all = np.reshape(np.array(y_list), (-1, 6))
        save_with_pickle(true_scores_all, filename_all, 'processed_folds', 'cmu_mosei')
    else:
        true_scores_all = load_from_pickle(filename_all, 'processed_folds', 'cmu_mosei')

    # Compute true presence scores: arrays of shape (4654, 6)
    # Possible values: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    # Coarse-grained values: [0, 1, 2, 3] if not prob else [0, 1/3, 2/3, 1]
    #### For training
    if not return_all_tasks and 'coarse' in label_type:
        if not pickle_file_exists(filename_coa_train, 'processed_folds', 'cmu_mosei'):
            true_scores_coa = get_coa_presence_score_for_training(true_scores_all, num_classes,
                                                                  prob=('prob' in label_type),
                                                                  one_hot=('oh' in label_type))
            save_with_pickle(true_scores_coa, filename_coa_train, 'processed_folds', 'cmu_mosei')
        else:
            true_scores_coa = load_from_pickle(filename_coa_train, 'processed_folds', 'cmu_mosei')
    #### For metric calculation
    if return_all_tasks:
        if not pickle_file_exists(filename_coa_met, 'processed_folds', 'cmu_mosei'):
            true_scores_coa = get_presence_score_for_metrics(true_scores_all, all_scores, num_classes, coarse=True)
            save_with_pickle(true_scores_coa, filename_coa_met, 'processed_folds', 'cmu_mosei')
        else:
            true_scores_coa = load_from_pickle(filename_coa_met, 'processed_folds', 'cmu_mosei')

    # Compute true present classes: binary arrays of shape (num_test, 6 or 7)
    if return_all_tasks or label_type == 'present':
        true_classes_pres = []
        for id, thres in enumerate(thres_score_pres):
            if not pickle_file_exists(filename_pres[id], 'processed_folds', 'cmu_mosei'):
                true_classes_pres_i = get_class_from_presence_score(true_scores_all, predict_neutral_class, [thres],
                                                                    None, num_classes, is_dominant=False,
                                                                    is_true_labels=True, label_type=label_type)
                true_classes_pres += true_classes_pres_i
                save_with_pickle(true_classes_pres_i, filename_pres[id], 'processed_folds', 'cmu_mosei')
            else:
                true_classes_pres += load_from_pickle(filename_pres[id], 'processed_folds', 'cmu_mosei')

    # Compute true dominant classes: binary arrays of shape (num_test, 6 or 7)
    if return_all_tasks or label_type == 'dominant':
        if not pickle_file_exists(filename_dom, 'processed_folds', 'cmu_mosei'):
            true_classes_dom = get_class_from_presence_score(true_scores_all, predict_neutral_class, None, None,
                                                             num_classes, is_dominant=True, is_true_labels=True,
                                                             label_type=label_type)
            save_with_pickle(true_classes_dom, filename_dom, 'processed_folds', 'cmu_mosei')
        else:
            true_classes_dom = load_from_pickle(filename_dom, 'processed_folds', 'cmu_mosei')

    if return_all_tasks:
        return true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom
    elif label_type == 'all':
        return true_scores_all
    elif 'coarse' in label_type:
        return true_scores_coa
    elif label_type == 'present':
        return true_classes_pres
    elif label_type == 'dominant':
        return true_classes_dom
    else:
        return ValueError("Value label_type invalid.")


def compute_pred_labels(pred_raw, all_scores, predict_neutral_class, thres_score_pres, thres_multilabel,
                        save_pred, model_id, num_classes, model_folder, label_type):
    """
    Compute the prediction labels (all arrays of shape (test_size, 6), pred_classes_pres is a list of arrays of this type)
    pred_scores_coa: Closest predicted score among [0, 1, 2, 3]
    pred_classes_pres: Emotions present predicted by the model (varying thresholds for presence score given by thres_score_pres)
    pred_classes_dom: Dominant emotions predicted by the model (highest presence score)

    :param pred_raw: array of shape (test_size, num_classes) predicting the presence score of the emotions (in [0,3])
    :param all_scores: list of available presence scores
    :param predict_neutral_class: Whether we predict the neutral class
    :param thres_score_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param save_pred: Whether we save predictions with pickle
    :param model_id: Model id (int)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param model_folder: The folder where the results will be saved
    :param label_type: Type of labels used for model training and evaluation.
    :return: pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom
    """

    # In the following lines, some labels do not make sense for the targeted task (label_type): we set them as None
    pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom = None, None, None, None
    if label_type == 'all':
        compute_fine, compute_coa, compute_pres, compute_dom = True, True, True, True
    elif 'coarse' in label_type:
        compute_fine, compute_coa, compute_pres, compute_dom = False, True, True, True
    elif label_type == 'present':
        compute_fine, compute_coa, compute_pres, compute_dom = False, False, True, False
    else:  # dominant
        compute_fine, compute_coa, compute_pres, compute_dom = False, False, False, True

    # Save raw predictions
    if save_pred:
        save_with_pickle(pred_raw, "pred_raw_{}".format(model_id), pickle_folder="predictions",
                         root_folder=model_folder)

    print("\n\nTargeted task:", label_type)

    # Get fine-grained presence scores from raw predictions (the closest fine-grained presence score)
    if compute_fine:
        pred_scores_all = get_presence_score_for_metrics(pred_raw, all_scores, num_classes, coarse=False)
        print("pred_scores_all computed: Closest presence score among those present in the original dataset.")
        if save_pred:
            save_with_pickle(pred_scores_all, "pred_scores_all_{}".format(model_id), pickle_folder="predictions",
                             root_folder=model_folder)

    # Get coarse presence scores from raw predictions (the closest coarse-grained presence score)
    if compute_coa:
        pred_scores_coa = get_presence_score_for_metrics(pred_raw, all_scores, num_classes, coarse=True)
        print("pred_scores_coa computed: Closest presence score among 0, 1, 2, 3.")
        if save_pred:
            save_with_pickle(pred_scores_coa, "pred_scores_coa_{}".format(model_id), pickle_folder="predictions",
                             root_folder=model_folder)

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

    return pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom


def update_labels(train_list, valid_list, test_list, param_task, all_scores):
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
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param all_scores: list of available presence scores
    :return: new_train_list, new_valid_list, new_test_list, with the same structure as the input, + neutral_test_ind
    """

    def remove_neutral_class(train_l, valid_l, test_l):
        """
        Remove samples assigned to neutral class + return the list of their indexes for test split

        :param train_l: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
        :param valid_l: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
        :param test_l: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
        :return: new_train_l, new_valid_l, new_test_l, neutral_test_ind_list
        """

        def remove_neutral_class_split(split_list, test=False):
            y_arr = np.vstack(split_list[1])
            num_emo = np.sum(y_arr, axis=1)
            non_neutral_ind = (num_emo != 0)  # Boolean array of shape (split_size,)
            num_neutral = np.sum((num_emo == 0))

            # Take only samples where at least one emotion class is assigned
            X_list = [elt for (elt, ind) in zip(split_list[0], non_neutral_ind) if ind]
            y_list = [elt for (elt, ind) in zip(split_list[1], non_neutral_ind) if ind]
            seg_list = [elt for (elt, ind) in zip(split_list[2], non_neutral_ind) if ind]
            neutral_test_index = [i for i, ind in enumerate(non_neutral_ind) if not ind] if test else None

            return [X_list, y_list, seg_list], num_neutral, neutral_test_index

        new_train_l, num_neutral_train, _ = remove_neutral_class_split(train_l)
        new_valid_l, num_neutral_valid, _ = remove_neutral_class_split(valid_l)
        new_test_l, num_neutral_test, neutral_test_ind_list = remove_neutral_class_split(test_l, test=True)

        num_neutral_samples = num_neutral_train + num_neutral_valid + num_neutral_test
        print(f'Removed {num_neutral_samples} samples with no emotions '
              f'(train: {num_neutral_train}, valid: {num_neutral_valid}, test: {num_neutral_test}).')

        return new_train_l, new_valid_l, new_test_l, neutral_test_ind_list

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

    label_type = param_task['label_type']
    predict_neutral_class = param_task['predict_neutral_class']
    thres_score_pres = [param_task['thres_score_pres']]
    thres_nan = param_task['thres_nan']
    num_classes = 7 if predict_neutral_class else 6

    y_train = train_list[1]
    y_valid = valid_list[1]
    y_test = test_list[1]

    new_y_train = compute_true_labels(y_train, "train", all_scores, predict_neutral_class, thres_score_pres,
                                      param_task, num_classes, label_type)
    new_y_valid = compute_true_labels(y_valid, "valid", all_scores, predict_neutral_class, thres_score_pres,
                                      param_task, num_classes, label_type)
    new_y_test = compute_true_labels(y_test, "test", all_scores, predict_neutral_class, thres_score_pres,
                                     param_task, num_classes, label_type)

    if label_type == 'present':  # TODO We only take the first threshold for the moment
        new_y_train = new_y_train[0]
        new_y_valid = new_y_valid[0]
        new_y_test = new_y_test[0]

    # Adapt to model training and evaluation
    new_y_train = change_type_and_shape_for_training(new_y_train)
    new_y_valid = change_type_and_shape_for_training(new_y_valid)
    new_y_test = change_type_and_shape_for_training(new_y_test)

    new_train_list = [train_list[0], new_y_train, train_list[2]]
    new_valid_list = [valid_list[0], new_y_valid, valid_list[2]]
    new_test_list = [test_list[0], new_y_test, test_list[2]]

    # When predict_neutral_class is False (no class created for Neutral),
    # samples with neutral class belong to any of available classes
    neutral_test_ind = None  # when predict_neutral_class is True
    if not predict_neutral_class:
        new_train_list, new_valid_list, new_test_list, neutral_test_ind = remove_neutral_class(new_train_list,
                                                                                               new_valid_list,
                                                                                               new_test_list)

    return new_train_list, new_valid_list, new_test_list, neutral_test_ind


def balance_dataset(train_list, valid_list, test_list, param_task, all_scores):
    """

    :param train_list: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
    :param valid_list: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
    :param test_list: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param all_scores: list of available presence scores
    :return:
    """

    label_type = param_task['label_type']
    predict_neutral_class = param_task['predict_neutral_class']
    thres_score_pres = [param_task['thres_score_pres']]
    ext_name = "_n" if predict_neutral_class else ""
    num_classes = 7 if predict_neutral_class else 6

    y_train = train_list[1]
    y_valid = valid_list[1]
    y_test = test_list[1]

    # TODO Find a way to balance emotional profiles

    # new_y_train = compute_true_labels(y_train, "train", all_scores, predict_neutral_class, thres_score_pres,
    #                                   num_classes, ext_name, "dominant")
    # new_y_valid = compute_true_labels(y_valid, "valid", all_scores, predict_neutral_class, thres_score_pres,
    #                                   num_classes, ext_name, "dominant")
    # new_y_test = compute_true_labels(y_test, "test", all_scores, predict_neutral_class, thres_score_pres, num_classes,
    #                                  ext_name, "dominant")

    # print("new_y_train dominant:", np.sum(new_y_train, axis=0))
    # print("new_y_valid dominant:", np.sum(new_y_valid, axis=0))
    # print("new_y_test dominant:", np.sum(new_y_test, axis=0))
    # print("new_y_train", np.unique(np.sum(new_y_train, axis=1), return_counts=True))
    # print("new_y_valid", np.unique(np.sum(new_y_valid, axis=1), return_counts=True))
    # print("new_y_test", np.unique(np.sum(new_y_test, axis=1), return_counts=True))
