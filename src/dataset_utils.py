import re
import numpy as np
import random
from mmsdk import mmdatasdk
from src.pickle_functions import *
from src.label_processing import trunc
from statistics import mean
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from scipy.interpolate import griddata


# CMU-MOSEI DATA DOWNLOAD FUNCTIONS

def avg_collapse_function(intervals: np.array, features: np.array) -> np.array:
    """
    Collapse function for data alignment using average for visual and vocal modality alignment to textual modality.
    It only averages modality when it is possible (not the case for text modality).
    """
    try:
        return np.average(features, axis=0)
    except:
        return features


def download_dataset(pickle_name_dataset, align_to_text):
    """
    Download CMU-MOSEI dataset using the SDK and perform data alignment (if desired).

    :param pickle_name_dataset: name of the pickle object that will contain the CMU-MOSEI mmdataset
    :param align_to_text: whether we want data to align to the textual modality
    """

    # The next two lines for safety. The function should be called only if the pickle obj does not exist
    if pickle_file_exists(pickle_name_dataset, pickle_folder="mmdataset", root_folder="cmu_mosei"):
        print("{} already exists in {} folder. Please change the pickle name.".format(pickle_name_dataset, "cmu_mosei"))

    else:
        if os.path.exists('cmu_mosei/'):
            # Retrieve data from cmu_mosei folder
            cmu_mosei = mmdatasdk.mmdataset('cmu_mosei/')
        else:
            # Download data and add them to cmu_mosei folder
            cmu_mosei = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, 'cmu_mosei/')

        if align_to_text:
            cmu_mosei.align('glove_vectors', collapse_functions=[avg_collapse_function])

        cmu_mosei.add_computational_sequences(mmdatasdk.cmu_mosei.labels, 'cmu_mosei/')
        cmu_mosei.align('All Labels')

        # Save cmu_mosei mmdataset with pickle
        save_with_pickle(cmu_mosei, pickle_name_dataset, pickle_folder="mmdataset", root_folder="cmu_mosei")


def get_dataset_from_sdk(param_log, param_task):
    """
    Return CMU-MOSEI mmdataset.

    :param param_log: dictionary of 'logistics' hyperparameters (cf. param_logistics.json)
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :return: CMU-MOSEI mmdataset
    """

    pickle_name_dataset = param_log['pickle_name_dataset']
    align_to_text = param_task['align_to_text']

    if not pickle_file_exists(pickle_name_dataset, pickle_folder="mmdataset", root_folder="cmu_mosei"):
        # Download from sdk and save with pickle
        download_dataset(pickle_name_dataset, align_to_text)

    # Get CMU-MOSEI mmdataset object from pickle
    dataset = load_from_pickle(pickle_name_dataset, pickle_folder='mmdataset', root_folder="cmu_mosei")
    print("CMU-MOSEI dataset loaded from pickle.")
    print("The existing computational sequences in dataset are: {}".format(list(dataset.keys())))

    return dataset


# CMU-MOSEI DATA SPLIT INTO TRAINING, VALIDATION AND TEST SETS

def perform_custom_split(train_ids, valid_ids, test_ids):
    """
    Perform custom split on training and validation sets.
    Final training set:   85% of original training data
    Final validation set: 10% of training data and 50% of validation data
    Final test set:       (5% of training data and 50% of validation data) + original test data

    :param train_ids: list of segment ids for training set
    :param valid_ids: list of segment ids for validation set
    :param test_ids: list of segment ids for test set
    :return: new train_ids, valid_ids, test_ids
    """

    def custom_split(train, valid):
        """
        Create three sets (training, validation and test) based on two sets (training and validation).
        Final training set:   85% of original training data
        Final validation set: 10% of training data and 50% of validation data
        Final test set:        5% of training data and 50% of validation data

        :param train: list of ids for training set
        :param valid: list of ids for validation set
        :return: 3 lists of ids for training, validation, and test sets respectively
        """
        total = len(valid)
        half = total // 2
        valid_ids_list = valid[:half]
        test_ids_list = valid[half + 1:]
        # 5 % of training into test data
        five_p = int(len(train) * 0.05)
        train_ids_list = train[:-five_p]
        test_ids_list = test_ids_list + train[-five_p:]
        # 10% of leftover training into valid data
        ten_p = int(len(train_ids_list) * 0.1)
        train_ids_list = train_ids_list[:-ten_p]
        valid_ids_list = valid_ids_list + train_ids_list[-ten_p:]
        return train_ids_list, valid_ids_list, test_ids_list

    train_ids_cs, valid_ids_cs, test_ids_cs = custom_split(train_ids, valid_ids)
    new_train_ids, new_valid_ids, new_test_ids = train_ids_cs, valid_ids_cs, test_ids_cs + test_ids

    return new_train_ids, new_valid_ids, new_test_ids


def get_fold_ids(param_task):
    """
    Get CMU-MOSEI standard fold ids for training, validation, and test

    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :return: 3 lists of ids for training, validation, and test sets respectively
    """

    with_custom_split = param_task['with_custom_split']

    train_ids = mmdatasdk.cmu_mosei.standard_folds.standard_train_fold
    valid_ids = mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold
    test_ids = mmdatasdk.cmu_mosei.standard_folds.standard_test_fold

    if with_custom_split:
        train_ids, valid_ids, test_ids = perform_custom_split(train_ids, valid_ids, test_ids)

    return train_ids, valid_ids, test_ids


def process_not_finite_values(data_seg, method):
    """
    Replace NaN and inf values by the method chosen by the user. If all values of a column are NaN/inf, replace by 0.

    :param data_seg: Array of shape (num_steps, num_features) that may contain NaN and/or inf values
    :param method: Method of replacement of NaN/inf values, among 'mean', 'linear_1d', 'linear_2d'
      - 'mean': replace all NaN/inf values by the mean of the feature column
      - 'linear_1d': 1D linear interpolation of the feature column, over the time axis
      - 'linear_2d': 2D linear interpolation when possible, else nearest neighbor (for the borders)
    :return: data_seg without any NaN/inf values
    """

    num_not_finite = np.sum(~np.isfinite(data_seg))
    if num_not_finite > 0:
        # Processing
        mean_not_finite = ~np.isfinite(np.sum(data_seg, axis=0))
        col_with_all_infinite = []
        for j in range(data_seg.shape[1]):
            data_j = np.copy(data_seg[:, j])
            if mean_not_finite[j]:
                if np.all(~np.isfinite(data_j)):
                    col_with_all_infinite += [j]
                else:
                    # Replace NaN/inf values with either linear interpolation or mean
                    if method == 'mean':
                        mean_j = np.mean(data_j[np.isfinite(data_j)])
                        for i in range(data_seg.shape[0]):
                            if not np.isfinite(data_seg[i, j]):
                                data_seg[i, j] = mean_j
                    elif method == 'linear_1d':
                        x = np.array(range(data_j.shape[0]))
                        # Linear interpolation
                        data_j[~np.isfinite(data_j)] = np.interp(x[~np.isfinite(data_j)],
                                                                 x[np.isfinite(data_j)], data_j[np.isfinite(data_j)])
                        data_seg[:, j] = data_j

        # Linear interpolation when possible, else nearest neighbor (for the borders)
        if method == 'linear_2d':
            x, y = np.indices(data_seg.shape)
            data_seg[~np.isfinite(data_seg)] = griddata((x[np.isfinite(data_seg)], y[np.isfinite(data_seg)]),
                                                        data_seg[np.isfinite(data_seg)],
                                                        (x[~np.isfinite(data_seg)], y[~np.isfinite(data_seg)]),
                                                        method='linear')

            data_seg[~np.isfinite(data_seg)] = griddata((x[np.isfinite(data_seg)], y[np.isfinite(data_seg)]),
                                                        data_seg[np.isfinite(data_seg)],
                                                        (x[~np.isfinite(data_seg)], y[~np.isfinite(data_seg)]),
                                                        method='nearest')
        num_steps = data_seg.shape[0]
        for j in col_with_all_infinite:
            data_seg[:, j] = np.array([0 for _ in range(num_steps)])

    return data_seg


def process_and_split_dataset(dataset, train_ids, valid_ids, test_ids, param_task, param_log, with_cross_val):
    """
    For each training, validation and test sets (and no split set if cross-validation), create a list of three sublists:
    - one for features (x): a list of 3 arrays of shape (number steps, number features) for text, image and audio
    features in this order
    - one for labels (y): arrays of shape (1, 6), for the 6 emotions
    - one for segment ids (seg): id of the segment described by (x, y). Example: 'zk2jTlAtvSU[1]'
    (Followed tutorial https://github.com/Justin1904/CMU-MultimodalSDK-Tutorials/blob/master/tutorial_interactive.ipynb)
    Note that this function performs **early fusion** (concatenation of low level text, image and audio features).

    :param dataset: CMU-MOSEI mmdataset
    :param train_ids: list of training ids
    :param valid_ids: list of validation ids
    :param test_ids: list of test ids
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param param_log: dictionary of 'logistics' hyperparameters (cf. param_logistics.json)
    :param with_cross_val: boolean (cross_validation or not)
    :return:  lists train_list = [x_train, y_train, seg_train], valid_list = [x_valid, y_valid, seg_valid],
     test_list = [x_test, y_test, seg_test], no_split_list = [x_no_split, y_no_split, seg_no_split]
    """

    image_feature = param_task['image_feature']
    thres_nan = param_task['thres_nan']
    thres_nan_100 = np.round(thres_nan*100)
    replace_nan = param_task['replace_nan']
    nan_str = replace_nan + "_tnan_" + str(thres_nan)
    length_seq = param_task['set_length']
    length_str = "_length_" + str(length_seq) if type(length_seq) == int else ""
    pickle_name_fold = param_log['pickle_name_fold']

    # Placeholders for training, validation, and test sets
    x_train, y_train, seg_train = [], [], []
    x_valid, y_valid, seg_valid = [], [], []
    x_test, y_test, seg_test = [], [], []
    if with_cross_val:
        x_no_split, y_no_split, seg_no_split = [], [], []

    image_feature_id = 'FACET 4.2' if image_feature == 'facet' else 'OpenFace_2'
    image_dataset = dataset[image_feature_id]
    audio_dataset = dataset['COVAREP']
    text_dataset = dataset['glove_vectors']
    label_dataset = dataset['All Labels']

    pattern = re.compile('(.*)\[.*\]')
    num_no_split = 0  # counter for those that doesn't belong to any split
    num_missing_mod, num_not_same_length, num_too_long_seq, num_nan = 0, 0, 0, 0

    for seg_id in dataset['All Labels'].keys():

        # Get video id and its image, audio, text and label features
        vid = re.search(pattern, seg_id).group(1)

        if not (seg_id in image_dataset.keys() and seg_id in audio_dataset.keys() and seg_id in text_dataset.keys()):
            print("Encountered datapoint {} that does not appear in image, audio, or text dataset.".format(seg_id))
            num_missing_mod += 1
            continue

        image_seg = image_dataset[seg_id]['features']
        audio_seg = audio_dataset[seg_id]['features']
        text_seg = text_dataset[seg_id]['features']
        label_seg = label_dataset[seg_id]['features']

        if not image_seg.shape[0] == audio_seg.shape[0] == text_seg.shape[0]:
            print("Encountered datapoint {} with image shape {}, audio shape {} and text shape {}."
                  .format(seg_id, image_seg.shape, audio_seg.shape, text_seg.shape))
            num_not_same_length += 1
            continue

        if type(length_seq) == int:
            if image_seg.shape[0] > length_seq:
                print("Encountered datapoint {} of length greater than {}.".format(seg_id, length_seq))
                num_too_long_seq += 1
                continue

        # Remove samples with percentage of NaN/inf above thres_nan
        shape_img = image_seg.shape[0] * image_seg.shape[1]
        shape_aud = audio_seg.shape[0] * audio_seg.shape[1]
        perc_not_finite_img = np.sum(~np.isfinite(image_seg)) / shape_img
        perc_not_finite_aud = np.sum(~np.isfinite(audio_seg)) / shape_aud
        if perc_not_finite_img > thres_nan or perc_not_finite_aud > thres_nan:
            print("Encountered datapoint {} with image data with more than {}% of NaNs or infs "
                  "in the original dataset.".format(seg_id, thres_nan_100))
            num_nan += 1
            continue

        # Non-finite value processing for image
        if perc_not_finite_img > 0:
            image_seg = process_not_finite_values(image_seg, replace_nan)

        # Non-finite value processing for audio
        if perc_not_finite_aud > 0:
            audio_seg = process_not_finite_values(audio_seg, replace_nan)

        # data_seg = np.concatenate((text_seg, image_seg, audio_seg), axis=1)  # early fusion

        if vid in train_ids:
            x_train.append([text_seg, image_seg, audio_seg])
            y_train.append(label_seg)
            seg_train.append(seg_id)
        elif vid in valid_ids:
            x_valid.append([text_seg, image_seg, audio_seg])
            y_valid.append(label_seg)
            seg_valid.append(seg_id)
        elif vid in test_ids:
            x_test.append([text_seg, image_seg, audio_seg])
            y_test.append(label_seg)
            seg_test.append(seg_id)
        else:
            if with_cross_val:
                x_no_split.append([text_seg, image_seg, audio_seg])
                y_no_split.append(label_seg)
                seg_no_split.append(seg_id)
            else:
                # print("Encountered video {} that does not belong to any split.".format(vid))
                num_no_split += 1

    train_size = len(y_train)
    valid_size = len(y_valid)
    test_size = len(y_test)
    if with_cross_val:
        no_split_size = len(y_no_split)
    # number of data points that doesn't go through the checking process
    num_drop = num_nan + num_missing_mod + num_not_same_length + num_too_long_seq

    print("----------------------------------------------------")
    print("Split dataset complete!")
    total_data_split = train_size + valid_size + test_size + no_split_size if with_cross_val else \
        train_size + valid_size + test_size
    total_data = total_data_split + num_drop + num_no_split
    print("Total number of {} datapoints have been dropped ({:.2f}% of total data)."
          .format(num_drop, 100 * (num_drop / total_data)))
    print("Among dropped datapoints, {} with missing modality ({:.2f}% of total data)."
          .format(num_missing_mod, 100 * (num_missing_mod / total_data)))
    print("Among dropped datapoints, {} with modalities with different length ({:.2f}% of total data)."
          .format(num_not_same_length, 100 * (num_not_same_length / total_data)))
    print("Among dropped datapoints, {} with at least {}% NaN or inf ({:.2f}% of total data)."
          .format(num_nan, thres_nan_100, 100 * (num_nan / total_data)))
    print("Among dropped datapoints, {} with sequence length > {} ({:.2f}% of total data)."
          .format(num_too_long_seq, length_seq, 100 * (num_too_long_seq / total_data)))
    if not with_cross_val:
        print("Total number of {} datapoints do not belong to any split ({:.2f}% of total data)."
              .format(num_no_split, 100 * (num_no_split / total_data)))
    print("----------------------------------------------------")
    print("Total number samples after checking step: {}".format(total_data_split))
    print("----------------------------------------------------")
    if not with_cross_val:
        print("Number of training datapoints: {} ({:.2f}%)".format(train_size, 100 * (train_size / total_data_split)))
        print("Number of validation datapoints: {} ({:.2f}%)".format(valid_size, 100 * (valid_size / total_data_split)))
        print("Number of test datapoints: {} ({:.2f}%)".format(test_size, 100 * (test_size / total_data_split)))

    # Remove sentiment labels + truncate to 2 decimals to obtain a non-continuous value
    y_train_emo = [trunc(y[:, 1:], decimals=2) for y in y_train]
    y_valid_emo = [trunc(y[:, 1:], decimals=2) for y in y_valid]
    y_test_emo = [trunc(y[:, 1:], decimals=2) for y in y_test]
    y_no_split_emo = [trunc(y[:, 1:], decimals=2) for y in y_no_split] if with_cross_val else None

    train_res = [x_train, y_train_emo, seg_train]
    valid_res = [x_valid, y_valid_emo, seg_valid]
    test_res = [x_test, y_test_emo, seg_test]
    no_split_res = [x_no_split, y_no_split_emo, seg_no_split] if with_cross_val else None

    # Save lists with pickle
    save_with_pickle(train_res, pickle_name_fold + "_train_" + nan_str + length_str,
                     pickle_folder="raw_folds", root_folder="cmu_mosei")
    save_with_pickle(valid_res, pickle_name_fold + "_valid_" + nan_str + length_str,
                     pickle_folder="raw_folds", root_folder="cmu_mosei")
    save_with_pickle(test_res, pickle_name_fold + "_test_" + nan_str + length_str,
                     pickle_folder="raw_folds", root_folder="cmu_mosei")
    if with_cross_val:
        save_with_pickle(no_split_res, pickle_name_fold + "_no_split_" + nan_str + length_str,
                         pickle_folder="raw_folds", root_folder="cmu_mosei")

    return train_res, valid_res, test_res, no_split_res


def split_for_cross_validation(train_list, valid_list, test_list, no_split_list, param_training, param_task,
                               data_ext_name):

    def create_split_for_one_fold(index):
        l_res = []
        x_new_fold = [X_dataset[i] for i in index]
        y_new_fold = [y_dataset[i] for i in index]
        seg_new_fold = [seg_dataset[i] for i in index]
        l_res += [x_new_fold, y_new_fold, seg_new_fold]
        return l_res

    def deal_with_single_class_strat(y_strat):
        """
        For all single y values, modify their number in base 2 by changing randomly 1 bit then 2, 3, ..., num_classes)
         and iteratively check if the new class in base 10 is present at least one time in y_strat (if so, assign the
         sample to this new class, else continue to find an appropriate class that meets this condition).
        :param y_strat: list of classes of type int (from 0 to 2^(num_classes)-1)
        :return: y_strat with no class with a single sample
        """
        num_classes = 6 if not param_task['predict_neutral_class'] else 7

        strat_cl, strat_nb = np.unique(y_strat, return_counts=True)
        for cl_i, nb_i in zip(strat_cl, strat_nb):
            if nb_i == 1:
                treated = False
                num_indexes_to_change = 0

                # convert cl_i back to base 2, and pad the beginning with 0, to create a binary number of length num_classes
                cl_i_bin = bin(cl_i)[2:].rjust(num_classes, '0')
                while not treated and num_indexes_to_change < num_classes:
                    num_indexes_to_change += 1
                    comb_classes = list(combinations(range(num_classes), num_indexes_to_change))
                    random.shuffle(comb_classes)
                    for indexes_to_change in comb_classes:
                        cl_i_bin_new = str(np.copy(cl_i_bin))
                        for i in indexes_to_change:
                            cl_i_bin_new = cl_i_bin_new[:i] + str(abs(1 - int(cl_i_bin_new[i]))) + cl_i_bin_new[i + 1:]
                        cl_i_new = int(cl_i_bin_new, 2)
                        if y_strat.count(cl_i_new) > 1:
                            index_old_cl_i = y_strat.index(cl_i)
                            y_strat[index_old_cl_i] = cl_i_new
                            treated = True
                            # print("{} ({}) became {} ({})".format(cl_i, cl_i_bin, cl_i_new, cl_i_bin_new))
                            break
        return y_strat

    num_folds = param_training['cross_val_folds']
    label_type = param_task['label_type']
    label_type_str = label_type[:4] if label_type == 'present' else label_type[:3]
    predict_neutral_class = param_task['predict_neutral_class']
    cv_filename = "cv_" + str(num_folds) + "_true_" + label_type_str + "_" + data_ext_name
    cv_filename += "_pred-neu" if predict_neutral_class else "_rm-neu"

    if pickle_file_exists("train_" + cv_filename, "processed_folds", "cmu_mosei"):
        train_res = load_from_pickle("train_" + cv_filename, 'processed_folds', "cmu_mosei")
        valid_res = load_from_pickle("valid_" + cv_filename, 'processed_folds', "cmu_mosei")
        test_res = load_from_pickle("test_" + cv_filename, 'processed_folds', "cmu_mosei")
        print("Folds for cross-validation with filename '{}' loaded from folder 'processed_folds'.".format(cv_filename))
    else:
        random_state = param_training['random_state']
        X_dataset = train_list[0] + valid_list[0] + test_list[0] + no_split_list[0]
        y_dataset = train_list[1] + valid_list[1] + test_list[1] + no_split_list[1]
        seg_dataset = train_list[2] + valid_list[2] + test_list[2] + no_split_list[2]

        # Use one-hot encoding labels (e.g. [0, 1, 0, 1, 1, 1] for stratification)
        if label_type in ['present', 'dominant']:
            y_strat_dataset = [list(y.flatten()) for y in y_dataset]
        else:
            thres_score_pres = param_task['thres_score_pres']
            y_strat_dataset = [list((y.flatten() > thres_score_pres).astype(int)) for y in y_dataset]

        # From one-hot labels (equivalent as number in base 2), generate class numbers in base 10
        y_strat_dataset = [int("".join(str(y_char) for y_char in y), 2) for y in y_strat_dataset]
        y_strat_dataset = deal_with_single_class_strat(y_strat_dataset)
        train_res, valid_res, test_res = [], [], []
        training_index_l, train_index_l, valid_index_l, test_index_l = [], [], [], []

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        for training_index, test_index in skf.split(X_dataset, y_strat_dataset):
            # Train and valid split, take 10% of training set for validation
            X_training = [X_dataset[i] for i in training_index]
            y_training_strat = [y_strat_dataset[i] for i in training_index]
            y_training_strat = deal_with_single_class_strat(y_training_strat)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
            train_index, valid_index = next(sss.split(X_training, y_training_strat))
            train_res += [create_split_for_one_fold(train_index)]
            valid_res += [create_split_for_one_fold(valid_index)]
            # Test split
            test_res += [create_split_for_one_fold(test_index)]
            training_index_l += [training_index]
            train_index_l += [train_index]
            valid_index_l += [valid_index]
            test_index_l += [test_index]
        save_with_pickle(train_res, "train_" + cv_filename, pickle_folder="processed_folds", root_folder="cmu_mosei")
        save_with_pickle(valid_res, "valid_" + cv_filename, pickle_folder="processed_folds", root_folder="cmu_mosei")
        save_with_pickle(test_res, "test_" + cv_filename, pickle_folder="processed_folds", root_folder="cmu_mosei")
        save_with_pickle([training_index_l, train_index_l, valid_index_l, test_index_l], "ind_" + cv_filename,
                         pickle_folder="processed_folds", root_folder="cmu_mosei")

    return train_res, valid_res, test_res


def data_normalization(train_list, valid_list, test_list, param_task, with_cross_val):
    """
    Normalize image and audio data from train_list, valid_list and test_list based on mean and std of train split.

    :param train_list: [X_train, y_train, seg_train]
    :param valid_list: [X_valid, y_valid, seg_valid]
    :param test_list: [X_test, y_test, seg_test]
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param with_cross_val: boolean (cross_validation or not)
    :return: new_train_list, new_valid_list, new_test_list
    """

    replace_nan = param_task['replace_nan']

    # This if/else block calculates mean and std of the whole training set for z-normalization
    if with_cross_val:
        img_array_train = [[X_img_seg for _, X_img_seg, _ in train_list[fold_id][0]] for fold_id in range(len(train_list))]
        aud_array_train = [[X_aud_seg for _, _, X_aud_seg in train_list[fold_id][0]] for fold_id in range(len(train_list))]
        img_mean_col_train = [np.mean(np.concatenate(img_arr), axis=0, keepdims=True) for img_arr in img_array_train]
        aud_mean_col_train = [np.mean(np.concatenate(aud_arr), axis=0, keepdims=True) for aud_arr in aud_array_train]
        img_std_col_train = [np.std(np.concatenate(img_arr), axis=0, keepdims=True) for img_arr in img_array_train]
        aud_std_col_train = [np.std(np.concatenate(aud_arr), axis=0, keepdims=True) for aud_arr in aud_array_train]
    else:
        img_array_train = [X_img_seg for _, X_img_seg, _ in train_list[0]]
        aud_array_train = [X_aud_seg for _, _, X_aud_seg in train_list[0]]
        img_mean_col_train = np.mean(np.concatenate(img_array_train), axis=0)
        aud_mean_col_train = np.mean(np.concatenate(aud_array_train), axis=0)
        img_std_col_train = np.std(np.concatenate(img_array_train), axis=0)
        aud_std_col_train = np.std(np.concatenate(aud_array_train), axis=0)

    def normalization_one_split(split_list, is_train):

        def normalization_modality(mod_list, mean_col, std_col):
            new_mod_list = []
            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 0

            for i, mod_array in enumerate(mod_list):
                # z-normalization along the feature column
                with np.errstate(invalid="ignore"):  # ignore errors which creates nan/infs, will be dealt right after
                    mod_array = (mod_array - mean_col) / (EPS + std_col)

                # After normalization, non-finite value processing for image and audio
                new_mod_array = process_not_finite_values(mod_array, replace_nan)
                new_mod_list += [new_mod_array]

            return new_mod_list

        if with_cross_val:
            new_x_split_list = []
            for fold_id, (i_mean_col, i_std_col, a_mean_col, a_std_col) in enumerate(zip(img_mean_col_train,
                                                                                         img_std_col_train,
                                                                                         aud_mean_col_train,
                                                                                         aud_std_col_train)):

                if is_train:  # 'train' already calculated earlier
                    img_list = img_array_train[fold_id]
                    aud_list = aud_array_train[fold_id]
                else:
                    img_list = [X_img_seg for _, X_img_seg, _ in split_list[fold_id][0]]
                    aud_list = [X_aud_seg for _, _, X_aud_seg in split_list[fold_id][0]]

                img_norm = normalization_modality(img_list, i_mean_col, i_std_col)
                aud_norm = normalization_modality(aud_list, a_mean_col, a_std_col)
                new_x_split_list += [[[split_list[fold_id][0][sample_id][0], X_img_i, X_aud_i]
                                     for sample_id, (X_img_i, X_aud_i) in enumerate(zip(img_norm, aud_norm))]]

            return [[new_x_split_list[fold_id], split_list[fold_id][1], split_list[fold_id][2]]
                    for fold_id in range(len(img_mean_col_train))]

        else:  # not with_cross_val
            img_list = [X_img_seg for _, X_img_seg, _ in split_list[0]] if not is_train else img_array_train
            aud_list = [X_aud_seg for _, _, X_aud_seg in split_list[0]] if not is_train else aud_array_train
            img_norm = normalization_modality(img_list, img_mean_col_train, img_std_col_train)
            aud_norm = normalization_modality(aud_list, aud_mean_col_train, aud_std_col_train)
            new_x_split_list = [[split_list[0][sample_id][0], X_img_i, X_aud_i] for sample_id, (X_img_i, X_aud_i) in
                                enumerate(zip(img_norm, aud_norm))]
            return [new_x_split_list, split_list[1], split_list[2]]

    new_train_list = normalization_one_split(train_list, is_train=True)
    new_valid_list = normalization_one_split(valid_list, is_train=False)
    new_test_list = normalization_one_split(test_list, is_train=False)

    return new_train_list, new_valid_list, new_test_list


def get_length_dataset(train_list, valid_list, test_list, param_task, with_cross_valid, is_tf):
    """
    Return the desired length for creating batches. (None if we want to keep variable lengths)
    :param train_list: [X_train, y_train, seg_train]
    :param valid_list: [X_valid, y_valid, seg_valid]
    :param test_list: [X_test, y_test, seg_test]
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param with_cross_valid: boolean (cross_validation or not)
    :param is_tf: whether it is TensorFlow or not (boolean)
    :return: Number of steps for each datapoint
    """

    set_length = param_task['set_length']

    def get_maximum_sequence_length_dataset():
        def get_maximum_sequence_length(x_list):
            res = 0
            # the length of the sequences for the three modalities should be the same
            # (checked in process_and_split_dataset)
            for x_i_text, _, _ in x_list:
                l_text = x_i_text.shape[0]
                res = max(l_text, res)
            return res
        if with_cross_valid:
            # Check the maximum sequence length in the first fold of each split is enough as we cover the whole dataset
            return max(get_maximum_sequence_length(train_list[0][0]), get_maximum_sequence_length(valid_list[0][0]),
                       get_maximum_sequence_length(test_list[0][0]))
        else:
            return max(get_maximum_sequence_length(train_list[0]), get_maximum_sequence_length(valid_list[0]),
                       get_maximum_sequence_length(test_list[0]))

    def get_average_sequence_length_dataset():
        def get_list_sequence_length(x_list):
            length_x_list = []
            # the length of the sequences for the three modalities should be the same
            # (checked in process_and_split_dataset)
            for x_i_text, _, _ in x_list:
                l_text = x_i_text.shape[0]
                length_x_list += [l_text]
            return length_x_list
        if with_cross_valid:
            # Check the average sequence length in the first fold of each split is enough as we cover the whole dataset
            all_length = get_list_sequence_length(train_list[0][0]) + get_list_sequence_length(valid_list[0][0]) + \
                         get_list_sequence_length(test_list[0][0])
        else:
            all_length = get_list_sequence_length(train_list[0]) + get_list_sequence_length(valid_list[0]) + \
                         get_list_sequence_length(test_list[0])
        return int(mean(all_length))

    if set_length == 'none':
        if is_tf:
            raise ValueError("Impossible to set set_length to 'none' for Tensorflow. Please choose value between: "
                             "'avg', 'max', any strictly positive number")
        return None
    elif set_length == 'max':
        return get_maximum_sequence_length_dataset()
    elif set_length == 'avg':
        return get_average_sequence_length_dataset()
    elif type(set_length) == int and int(set_length) > 0:
        return int(set_length)
    else:
        raise ValueError("Invalid value for parameter set_length. Please choose value between: 'avg', 'max',"
                         " any strictly positive number, 'none'")






