import numpy as np

from src.pickle_functions import *

from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer


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
    all_y_uniq_rounded = np.around(all_y_uniq, decimals=2)

    return list(all_y_uniq_rounded)


def get_presence_score_from_finer_grained_val(scores_array, all_scores, num_classes, coarse=False, prob=False,
                                              one_hot=False, pred=False):
    """
    Get the presence score from a finer grained presence score
    (raw predictions -> 12 or 4 presence scores, or 12 presence scores -> 4 presence scores)

    :param scores_array: input array of shape (test_size, num_classes) giving the presence score of the 6 emotions
    :param all_scores: list of available presence scores
    :param num_classes: number of emotion classes (7 with neutral class, else 6)
    :param coarse: if True, the resulting presence scores in [0, 1, 2, 3].
                   Default: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    :param prob: whether we turn coarse scores to probabilities [0, 1/3, 2/3, 1]
    :param one_hot: whether the output score is one hot encoded
    :param pred: whether the input array is obtained by prediction or not (bool)
    :return: array of shape (test_size, num_classes) [or (test_size, num_classes, num_scores) if one_hot] giving the
    presence score of all the 6 emotions
    """

    # True labels OR prediction & not one hot
    if not pred or (pred and not one_hot):
        scores = all_scores if not coarse else [0, 1, 2, 3] if not prob else [0, 1/3, 2/3, 1]
        scores_array = scores_array.flatten()

        # The following line returns the class closest to the input value
        out_scores = [min(list(scores), key=lambda x: abs(x - scores_array[i])) for i in range(scores_array.shape[0])]

        if one_hot:
            lb = LabelBinarizer()
            _ = lb.fit(scores)
            out_scores = lb.transform(out_scores)
            out_scores = np.reshape(np.array(out_scores), (-1, num_classes, len(scores)))
        else:
            out_scores = np.reshape(np.array(out_scores), (-1, num_classes))

    # Prediction on one_hot, keep in one-hot format
    else:
        # Check if the input score array is in one hot format
        if scores_array.shape[-1] != 4 or scores_array.shape[-2] != num_classes:
            raise ValueError("The output of the model is not in shape (num_samples, num_classes, num_scores=4)"
                             " as the label type requires one-hot encoding. Please check your model output.")

        out_scores = []
        for sample_i in range(scores_array.shape[0]):
            for class_j in range(num_classes):
                max_ind_ij = np.argmax(scores_array[sample_i, class_j])  # array (4,) for a given sample and class
                # For a given sample and class, loop over scores and assign 1 if maximum value else 0
                out_scores += [[int(k == max_ind_ij) for k in range(4)]]
        out_scores = np.array(out_scores)

    return out_scores


def get_class_from_presence_score(score_array, with_neutral_class, threshold_emo_pres, num_classes, only_dominant=False,
                                  pred=False):
    """
    Get the dominant emotion class(es) for each test segment based on the presence scores of emotions.
    If only_dominant, return a binary array, else return a list of len(threshold_emo_pres) binary arrays.
    Column indexes: 0:happy, 1:sad, 2:anger, 3:surprise, 4:disgust, 5:fear, (6:neutral)

    :param score_array: array of shape (test_size, num_classes) containing the presence score of 6 emotions (+ neutral)
    :param with_neutral_class: whether we add a neutral class or not
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param only_dominant: if True, only keep dominant classes (else keep any emotion present)
    :return: Binary array of shape (test_size,7) if we add neutral class, else array of shape (test_size, 6).
    """

    def get_list_emotions_for_each_seg(thres=None):
        """
        Get the list of all emotion ids for each segment.
        For example, if sad is index 1 and i-th segment only shows sadness, then the i-th element of the list is [1].
        If j-th segment shows no emotion, then j-th element of the list is [6] if we predict neutral class else [].
        :param thres: Threshold above which the emotion is considered as being present.
        :return: list of lists of emotion ids. The number of elements of this list is the number of segments.
        """
        list_emotions = []
        test_size = score_array.shape[0]

        for seg_id in range(test_size):  # we check every segment

            score_seg = score_array[seg_id, :]

            if pred:  # For predicted labels
                tmp = [i for i, val in enumerate(score_seg) if val > 0.5]
                tmp = [6] if len(tmp) == 0 and with_neutral_class else tmp
                list_emotions.append(tmp)

            else:  # For true labels
                max_score = max(score_seg)
                if max_score == 0:  # the current segment shows neutral (no emotions)
                    list_emotions.append([6]) if with_neutral_class else list_emotions.append([])
                elif only_dominant:
                    # the following takes the index(es) of the maximum value in the presence score vector of emotions
                    list_emotions.append([i for i, val in enumerate(score_seg) if val == max_score])
                else:
                    # For emotion present
                    tmp = [i for i, val in enumerate(score_seg) if val > thres]
                    tmp = [6] if len(tmp) == 0 and with_neutral_class else tmp  # just for safety
                    list_emotions.append(tmp)

        return list_emotions

    # Transform list of indexes to one-hot vector
    mlb = MultiLabelBinarizer(classes=list(range(num_classes)))

    if only_dominant:
        list_emotions = get_list_emotions_for_each_seg()
        bin_array_emotions = mlb.fit_transform(list_emotions)
        return bin_array_emotions

    else:  # Collect all emotions present in the sample
        list_emotions_all_thresholds = []

        for thres in threshold_emo_pres:
            list_emotions_thres = get_list_emotions_for_each_seg(thres=thres)
            list_emotions_all_thresholds.append(list_emotions_thres)

        list_bin_array_emotions = [mlb.fit_transform(l) for l in list_emotions_all_thresholds]
        return list_bin_array_emotions


def compute_true_labels(y_list, root_name, all_scores, predict_neutral_class, threshold_emo_pres, num_classes,
                        label_type, return_all_tasks=False):
    """
    Compute the true labels from a list of 2D arrays of shape (1, num_classes) (y_list)
    The results are all 3D arrays, true_classes_pres is a list of arrays of this type)
    true_scores_all: Presence scores provided by the dataset. Shape: (number of y elements, 6)
    true_scores_coa: Closest true score among [0, 1, 2, 3]. Shape: (number of y elements, 6)
    true_classes_pres: Emotions truly present (varying thresholds for presence score given by threshold_emo_pres).
    Shape: (number of y elements, num_classes)
    true_classes_dom: Emotions truly dominant (highest presence score). Shape: (number of y elements, num_classes)

    :param y_list: List of arrays of shape (1, num_classes)
    :param root_name: Name to append to pickle file (ex: 'train', 'valid', 'test')
    :param all_scores: list of available presence scores
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param num_classes: number of emotion classes (7 with neutral class, else 6)
    :param label_type: Type of labels used for model training and evaluation. Possible values: all (for all presence
                         scores provided by the dataset), coarse (presence score for each emotion among [0, 1, 2, 3]),
                         coarse_prob (presence score for each emotion among [0, 1/3, 2/3, 1]),
                         present (multi-label classification), dominant (multi-label classification)
    :param return_all_tasks: Whether we return the true vectors of the four tasks (True) or return the vector
                               corresponding to the label type (False)
    :return: true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom depending on return_all_tasks
    """

    ext_name = "_pred-neu" if predict_neutral_class else "_rm-neu"
    true_scores_coa, true_classes_pres, true_classes_dom = None, None, None

    # if not pickle_file_exists(root_name + "_true_scores_all" + ext_name, 'processed_folds', 'cmu_mosei'):
        
    # Create 2D array from list of 2D true presence score arrays of shape (1, 6)
    if not pickle_file_exists(root_name + "_true_scores_all" + ext_name, 'processed_folds', 'cmu_mosei'):
        true_scores_all = np.reshape(np.array(y_list), (-1, 6))
        save_with_pickle(true_scores_all, root_name + "_true_scores_all" + ext_name, 'processed_folds', 'cmu_mosei')
    else:
        true_scores_all = load_from_pickle(root_name + "_true_scores_all" + ext_name, 'processed_folds', 'cmu_mosei')

    # Compute true presence scores: arrays of shape (4654, 6)
    # Possible values: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    # Coarse-grained values: [0, 1, 2, 3] if not prob else [0, 1/3, 2/3, 1]
    if return_all_tasks or 'coarse' in label_type:
        true_scores_coa = get_presence_score_from_finer_grained_val(true_scores_all, all_scores, 6,
                                                                    coarse=True, prob=('prob' in label_type),
                                                                    one_hot=('oh' in label_type))
        save_with_pickle(true_scores_coa, root_name + "_true_scores_coa" + ext_name, 'processed_folds', 'cmu_mosei')

    # Compute true classes: binary arrays of shape (4654, 6 or 7)
    if return_all_tasks or label_type == 'present':
        true_classes_pres = get_class_from_presence_score(true_scores_all, predict_neutral_class,
                                                          threshold_emo_pres, num_classes)
        save_with_pickle(true_classes_pres, root_name + "_true_classes_pres" + ext_name, 'processed_folds', 'cmu_mosei')

    if return_all_tasks or label_type == 'dominant':
        true_classes_dom = get_class_from_presence_score(true_scores_all, predict_neutral_class,
                                                         threshold_emo_pres, num_classes, only_dominant=True)
        save_with_pickle(true_classes_dom, root_name + "_true_classes_dom" + ext_name, 'processed_folds', 'cmu_mosei')

    # else:
    #     true_scores_all = load_from_pickle(root_name + "_true_scores_all" + ext_name, 'processed_folds', 'cmu_mosei')
    #     true_scores_coa = load_from_pickle(root_name + "_true_scores_coa" + ext_name, 'processed_folds', 'cmu_mosei')
    #     true_classes_pres = load_from_pickle(root_name + "_true_classes_pres" + ext_name, 'processed_folds', 'cmu_mosei')
    #     true_classes_dom = load_from_pickle(root_name + "_true_classes_dom" + ext_name, 'processed_folds', 'cmu_mosei')

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


def compute_pred_labels(pred_raw, all_scores, predict_neutral_class, threshold_emo_pres, save_pred, model_id,
                        num_classes, model_folder, label_type):
    """
    Compute the prediction labels (all arrays of shape (test_size, 6), pred_classes_pres is a list of arrays of this type)
    pred_scores_coa: Closest predicted score among [0, 1, 2, 3]
    pred_classes_pres: Emotions present predicted by the model (varying thresholds for presence score given by threshold_emo_pres)
    pred_classes_dom: Dominant emotions predicted by the model (highest presence score)

    :param pred_raw: array of shape (test_size, num_classes) predicting the presence score of the emotions
    :param all_scores: list of available presence scores
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param save_pred: Whether we save predictions with pickle
    :param model_id: Model id (int)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param model_folder: The folder where the results will be saved
    :param label_type: Type of labels used for model training and evaluation.
    :return: pred_scores_coa, pred_classes_pres, pred_classes_dom
    """

    # In the following lines, some labels do not make sense for the targeted task (label_type): we set them as None
    pred_scores_coa, pred_classes_pres = None, None
    if label_type == 'all':
        compute_coa, compute_pres = True, True
    elif 'coarse' in label_type:
        compute_coa, compute_pres = True, True
    elif label_type == 'present':
        compute_coa, compute_pres = False, True
    else:  # dominant
        compute_coa, compute_pres = False, False

    # Save raw predictions
    if save_pred:
        save_with_pickle(pred_raw, "pred_raw_{}".format(model_id), pickle_folder="predictions",
                         root_folder=model_folder)

    # # Get the closest presence score from raw prediction (useful for getting the other metrics)
    # pred_scores_all = get_presence_score_from_finer_grained_val(pred_raw, all_scores, num_classes)

    # Get coarse presence scores from raw predictions
    if compute_coa:
        pred_scores_coa = get_presence_score_from_finer_grained_val(pred_raw, all_scores, num_classes,
                                                                    coarse=True, prob=('prob' in label_type),
                                                                    one_hot=('oh' in label_type), pred=True)
        if save_pred:
            save_with_pickle(pred_scores_coa, "pred_scores_coa_{}".format(model_id), pickle_folder="predictions",
                             root_folder=model_folder)

    # Get present emotion classes predicted by the model, based on thresholds
    if compute_pres:
        pred_classes_pres = get_class_from_presence_score(pred_raw, predict_neutral_class, threshold_emo_pres,
                                                          num_classes, pred=True)
        if save_pred:
            save_with_pickle(pred_classes_pres, "pred_classes_pres_{}".format(model_id), pickle_folder="predictions",
                             root_folder=model_folder)

    # Get dominant emotion classes predicted by the model
    pred_classes_dom = get_class_from_presence_score(pred_raw, predict_neutral_class, threshold_emo_pres,
                                                     num_classes, only_dominant=True, pred=True)
    if save_pred:
        save_with_pickle(pred_classes_dom, "pred_classes_dom_{}".format(model_id), pickle_folder="predictions",
                         root_folder=model_folder)

    return pred_scores_coa, pred_classes_pres, pred_classes_dom


def update_labels(train_list, valid_list, test_list, param_task, all_scores):
    """
    Update labels for each fold so that it matches with the targeted task given by label_type:
    - all: original presence scores provided by the dataset
    - coarse: presence score for each emotion among [0, 1, 2, 3]
    - coarse_prob: presence score for each emotion among [0, 1/3, 2/3, 1]
    - present: one-hot encoded labels for multi-label classification. 7 classes if predict_neutral_class else 6.
    - dominant: one-hot encoded labels for one-label classification. 7 classes if predict_neutral_class else 6.
    It also removes samples assigned to neutral when predict_neutral_class is False.

    :param train_list: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
    :param valid_list: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
    :param test_list: [x, y, seg] with x a list of arrays of shape (nb_steps, nb_features), y a list of arrays of shape (1, 6), seg a list of strings
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param all_scores: list of available presence scores
    :return: new_train_list, new_valid_list, new_test_list, with the same structure as the input
    """

    def remove_neutral_class():

        def remove_neutral_class_split(split_list):
            y_arr = np.vstack(split_list[1])
            num_emo = np.sum(y_arr, axis=1)
            non_neutral_ind = (num_emo != 0)  # Boolean array of shape (split_size,)
            neutral_samples_split = np.sum((num_emo == 0))
            # Take only samples where at least one emotion class is assigned
            X_list = [elt for (elt, ind) in zip(split_list[0], non_neutral_ind) if ind]
            y_list = [elt for (elt, ind) in zip(split_list[1], non_neutral_ind) if ind]
            seg_list = [elt for (elt, ind) in zip(split_list[2], non_neutral_ind) if ind]
            return [X_list, y_list, seg_list], neutral_samples_split

        new_train_l, neutral_train = remove_neutral_class_split(train_list)
        new_valid_l, neutral_valid = remove_neutral_class_split(valid_list)
        new_test_l, neutral_test = remove_neutral_class_split(test_list)

        neutral_samples = neutral_train + neutral_valid + neutral_test
        print(f'Removed {neutral_samples} samples with no emotions '
              f'(train: {neutral_train}, valid: {neutral_valid}, test: {neutral_test}).')

        return new_train_l, new_valid_l, new_test_l

    def change_type_and_shape_for_training(y_array):
        num_output = y_array.shape[-1]
        y_list = list(y_array)
        y_list_res = [y.reshape((-1, num_output)) for y in y_list]
        return y_list_res

    label_type = param_task['label_type']
    predict_neutral_class = param_task['predict_neutral_class']
    threshold_emo_pres = [param_task['threshold_emo_present']]
    num_classes = 7 if predict_neutral_class else 6

    # When predict_neutral_class is False (no class created for Neutral),
    # samples with neutral class belong to any of available classes
    if not predict_neutral_class:
        train_list, valid_list, test_list = remove_neutral_class()

    y_train = train_list[1]
    y_valid = valid_list[1]
    y_test = test_list[1]

    new_y_train = compute_true_labels(y_train, "train", all_scores, predict_neutral_class, threshold_emo_pres, num_classes, label_type)
    new_y_valid = compute_true_labels(y_valid, "valid", all_scores, predict_neutral_class, threshold_emo_pres, num_classes, label_type)
    new_y_test = compute_true_labels(y_test, "test", all_scores, predict_neutral_class, threshold_emo_pres, num_classes, label_type)

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

    return new_train_list, new_valid_list, new_test_list


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
    threshold_emo_pres = [param_task['threshold_emo_present']]
    ext_name = "_n" if predict_neutral_class else ""
    num_classes = 7 if predict_neutral_class else 6

    y_train = train_list[1]
    y_valid = valid_list[1]
    y_test = test_list[1]

    # TODO Find a way to balance emotional profiles

    # new_y_train = compute_true_labels(y_train, "train", all_scores, predict_neutral_class, threshold_emo_pres,
    #                                   num_classes, ext_name, "dominant")
    # new_y_valid = compute_true_labels(y_valid, "valid", all_scores, predict_neutral_class, threshold_emo_pres,
    #                                   num_classes, ext_name, "dominant")
    # new_y_test = compute_true_labels(y_test, "test", all_scores, predict_neutral_class, threshold_emo_pres, num_classes,
    #                                  ext_name, "dominant")

    # print("new_y_train dominant:", np.sum(new_y_train, axis=0))
    # print("new_y_valid dominant:", np.sum(new_y_valid, axis=0))
    # print("new_y_test dominant:", np.sum(new_y_test, axis=0))
    # print("new_y_train", np.unique(np.sum(new_y_train, axis=1), return_counts=True))
    # print("new_y_valid", np.unique(np.sum(new_y_valid, axis=1), return_counts=True))
    # print("new_y_test", np.unique(np.sum(new_y_test, axis=1), return_counts=True))


