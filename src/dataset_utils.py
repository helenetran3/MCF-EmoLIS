import re
import numpy as np
from mmsdk import mmdatasdk
from src.pickle_functions import *
from src.label_processing import trunc
from statistics import mean
from sklearn.model_selection import KFold
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
    Replace NaN and inf values by the method chosen by the user.

    :param data_seg: Array of shape (num_steps, num_features) that may contain NaN and/or inf values
    :param method: Method of replacement of NaN/inf values, among 'mean', 'linear_1d', 'linear_2d'
      - 'mean': replace all NaN/inf values by the mean of the feature column
      - 'linear_1d': 1D linear interpolation of the feature column, over the time axis
      - 'linear_2d': 2D linear interpolation when possible, else nearest neighbor (for the borders)
    :return: data_seg without any NaN/inf values
    """

    one_col_with_all_infinite = 0
    num_col_with_half_nan = 0
    max_perc_nan_col = 0

    num_not_finite = np.sum(~np.isfinite(data_seg))
    if num_not_finite > 0:
        # Processing
        mean_not_finite = ~np.isfinite(np.sum(data_seg, axis=0))
        col_with_all_infinite = []
        for j in range(data_seg.shape[1]):
            data_j = np.copy(data_seg[:, j])
            if mean_not_finite[j]:
                if np.all(~np.isfinite(data_j)):
                    one_col_with_all_infinite += 1
                    col_with_all_infinite += [j]
                else:
                    # Compute percentage of not finite values in the column
                    perc_nan_col = np.sum(~np.isfinite(data_j)) / data_j.shape[0]
                    max_perc_nan_col = max(perc_nan_col, max_perc_nan_col)
                    if np.sum(~np.isfinite(data_j)) / data_j.shape[0] > 0.4:
                        num_col_with_half_nan += 1

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

    return data_seg, one_col_with_all_infinite, num_col_with_half_nan, max_perc_nan_col


def process_and_split_dataset(dataset, train_ids, valid_ids, test_ids, param_task, param_log):
    """
    For each training, validation and test sets, create a list of three sublists:
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
    :return: 3 lists train_list = [x_train, y_train, seg_train], valid_list = [x_valid, y_valid, seg_valid],
     test_list = [x_test, y_test, seg_test]
    """

    image_feature = param_task['image_feature']
    thres_nan = param_task['thres_nan']
    thres_nan_100 = np.round(thres_nan*100)
    replace_nan = param_task['replace_nan']
    nan_str = replace_nan + "_t_" + str(thres_nan)
    pickle_name_fold = param_log['pickle_name_fold']

    # Placeholders for training, validation, and test sets
    x_train, y_train, seg_train = [], [], []
    x_valid, y_valid, seg_valid = [], [], []
    x_test, y_test, seg_test = [], [], []

    image_feature_id = 'FACET 4.2' if image_feature == 'facet' else 'OpenFace_2'
    image_dataset = dataset[image_feature_id]
    audio_dataset = dataset['COVAREP']
    text_dataset = dataset['glove_vectors']
    label_dataset = dataset['All Labels']

    pattern = re.compile('(.*)\[.*\]')
    num_drop = 0  # counter to get the number of data points that doesn't go through the checking process
    num_no_split = 0  # counter for those that doesn't belong to any split
    num_some_nan_bf_after_imput_img, num_some_nan_bf_after_imput_aud = 0, 0
    img_one_col_with_all_infinite_bf, aud_one_col_with_all_infinite_bf = 0, 0
    img_one_col_with_half_infinite_bf, aud_one_col_with_half_infinite_bf = 0, 0
    num_all_nan, num_perc_nan = 0, 0
    max_percentage_nan_col_img_bf, max_percentage_nan_col_aud_bf = 0, 0
    num_nan_labels = 0
    num_nan_text = 0

    for seg_id in dataset['All Labels'].keys():

        # Get video id and its image, audio, text and label features
        vid = re.search(pattern, seg_id).group(1)

        if not (seg_id in image_dataset.keys() and seg_id in audio_dataset.keys() and seg_id in text_dataset.keys()):
            print("Encountered datapoint {} that does not appear in image, audio, or text dataset.".format(seg_id))
            num_drop += 1
            continue

        image_seg = image_dataset[seg_id]['features']
        audio_seg = audio_dataset[seg_id]['features']
        text_seg = text_dataset[seg_id]['features']
        label_seg = label_dataset[seg_id]['features']

        if not image_seg.shape[0] == audio_seg.shape[0] == text_seg.shape[0]:
            print("Encountered datapoint {} with image shape {}, audio shape {} and text shape {}."
                  .format(seg_id, image_seg.shape, audio_seg.shape, text_seg.shape))
            num_drop += 1
            continue

        if image_seg.shape[0] == 1:
            print("Encountered datapoint {} of length 1.".format(seg_id))
            num_drop += 1
            continue

        # Remove samples with percentage of NaN/inf above thres_nan
        shape_img = image_seg.shape[0] * image_seg.shape[1]
        shape_aud = audio_seg.shape[0] * audio_seg.shape[1]
        perc_not_finite_img = np.sum(~np.isfinite(image_seg)) / shape_img
        perc_not_finite_aud = np.sum(~np.isfinite(audio_seg)) / shape_aud
        if perc_not_finite_img >= thres_nan or perc_not_finite_aud >= thres_nan:
            print("Encountered datapoint {} with image data with more than {}% of NaNs or infs "
                  "in the original dataset.".format(seg_id, thres_nan_100))
            num_perc_nan += 1
            num_drop += 1
            continue

        if np.sum(~np.isfinite(text_seg)) > 0:
            num_nan_text += 1
        if np.sum(~np.isfinite(label_seg)) > 0:
            num_nan_labels += 1

        # Non-finite value processing for image
        if perc_not_finite_img > 0:
            image_seg, num_img2, num_img4, num_img5 = process_not_finite_values(image_seg, replace_nan)
            # For printing
            img_one_col_with_all_infinite_bf += num_img2
            img_one_col_with_half_infinite_bf += num_img4
            max_percentage_nan_col_img_bf = max(num_img5, max_percentage_nan_col_img_bf)
            perc_not_finite_img = np.sum(~np.isfinite(image_seg)) / shape_img
            if perc_not_finite_img > 0:
                num_some_nan_bf_after_imput_img += 1

        # Non-finite value processing for audio
        if perc_not_finite_aud > 0:
            audio_seg, num_aud2, num_aud4, num_aud5 = process_not_finite_values(audio_seg, replace_nan)
            # For printing
            aud_one_col_with_all_infinite_bf += num_aud2
            aud_one_col_with_half_infinite_bf += num_aud4
            max_percentage_nan_col_aud_bf = max(num_aud5, max_percentage_nan_col_aud_bf)
            perc_not_finite_aud = np.sum(~np.isfinite(audio_seg)) / shape_aud
            if perc_not_finite_aud > 0:
                num_some_nan_bf_after_imput_aud += 1

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
            # print("Encountered video {} that does not belong to any split.".format(vid))
            num_no_split += 1

    train_size = len(y_train)
    valid_size = len(y_valid)
    test_size = len(y_test)

    print("----------------------------------------------------")
    print("Split dataset complete!")
    total_data_split = train_size + valid_size + test_size
    total_data = total_data_split + num_drop + num_no_split
    print("Total number of {} datapoints have been dropped ({:.2f}% of total data)."
          .format(num_drop, 100 * (num_drop / total_data)))
    print("Total number of {} datapoints do not belong to any split ({:.2f}% of total data)."
          .format(num_no_split, 100 * (num_no_split / total_data)))
    print("Total number of dropped datapoints with at least {}% NaN or inf: {}".format(thres_nan_100, num_perc_nan))
    print("----------------------------------------------------")
    print("Total number samples: {}".format(total_data))
    print("Number of text data with NaN/inf:", num_nan_text)
    print("Number of label data with NaN/inf:", num_nan_labels)
    print("----------------------------------------------------")
    # print("[Image, before processing] Number of data with some NaN or inf: {}".format(num_some_nan_bf_img))
    print("[Image, before processing] Number of columns with all NaN or inf: {}".format(img_one_col_with_all_infinite_bf))
    print("[Image, before processing] Number of columns with at least 40% NaN or inf: {}".format(img_one_col_with_half_infinite_bf))
    print("[Image, before processing] Max percentage columns with NaN or inf: {}%".format(np.round(max_percentage_nan_col_img_bf*100, decimals=2)))
    print("[Image, after processing] Number of data with some NaN or inf: {}".format(num_some_nan_bf_after_imput_img))
    # print("\n[Audio, before processing] Number of data with some NaN or inf: {}".format(num_some_nan_bf_aud))
    print("[Audio, before processing] Number of columns with all NaN or inf: {}".format(aud_one_col_with_all_infinite_bf))
    print("[Audio, before processing] Number of columns with at least 40% NaN or inf: {}".format(aud_one_col_with_half_infinite_bf))
    print("[Audio, before processing] Max percentage columns with NaN or inf: {}%".format(np.round(max_percentage_nan_col_aud_bf*100, decimals=2)))
    print("[Audio, after processing] Number of data with some NaN or inf: {}".format(num_some_nan_bf_after_imput_aud))
    print("----------------------------------------------------")
    print("Number of training datapoints: {} ({:.2f}%)".format(train_size, 100 * (train_size / total_data_split)))
    print("Number of validation datapoints: {} ({:.2f}%)".format(valid_size, 100 * (valid_size / total_data_split)))
    print("Number of test datapoints: {} ({:.2f}%)".format(test_size, 100 * (test_size / total_data_split)))

    # Remove sentiment labels + truncate to 2 decimals to obtain a non-continuous value
    y_train_emo = [trunc(y[:, 1:], decimals=2) for y in y_train]
    y_valid_emo = [trunc(y[:, 1:], decimals=2) for y in y_valid]
    y_test_emo = [trunc(y[:, 1:], decimals=2) for y in y_test]

    train_res = [x_train, y_train_emo, seg_train]
    valid_res = [x_valid, y_valid_emo, seg_valid]
    test_res = [x_test, y_test_emo, seg_test]

    # Save lists with pickle
    save_with_pickle(train_res, pickle_name_fold + "_train_" + nan_str, pickle_folder="raw_folds", root_folder="cmu_mosei")
    save_with_pickle(valid_res, pickle_name_fold + "_valid_" + nan_str, pickle_folder="raw_folds", root_folder="cmu_mosei")
    save_with_pickle(test_res, pickle_name_fold + "_test_" + nan_str, pickle_folder="raw_folds", root_folder="cmu_mosei")

    return train_res, valid_res, test_res


def split_for_cross_validation(train_list, valid_list, param_training):

    def create_split_for_one_fold(index):
        l_res = []
        x_new_fold = [X_for_training[i] for i in index]
        y_new_fold = [y_for_training[i] for i in index]
        seg_new_fold = [seg_for_training[i] for i in index]
        l_res += [x_new_fold, y_new_fold, seg_new_fold]
        return l_res

    num_folds = param_training['cross_val_folds']
    random_state = param_training['random_state']

    X_for_training = train_list[0] + valid_list[0]
    y_for_training = train_list[1] + valid_list[1]
    seg_for_training = train_list[2] + valid_list[2]

    train_res, valid_res = [], []

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    for train_index, valid_index in kf.split(X_for_training):
        train_res += [create_split_for_one_fold(train_index)]
        valid_res += [create_split_for_one_fold(valid_index)]

    return train_res, valid_res


def data_normalization(train_list, valid_list, test_list, param_task, with_cross_val):
    """

    :param train_list: [X_train, y_train, seg_train]
    :param valid_list: [X_valid, y_valid, seg_valid]
    :param test_list: [X_test, y_test, seg_test]
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param with_cross_val: boolean (cross_validation or not)
    :return:
    """

    replace_nan = param_task['replace_nan']

    # This if/else block calculates mean and std of the whole training set for z-normalization
    if with_cross_val:
        img_mean_col_train, aud_mean_col_train = [], []
        img_std_col_train, aud_std_col_train = [], []
        img_array_train = [[X_img_seg for _, X_img_seg, _ in tl[0]] for tl in train_list]
        aud_array_train = [[X_aud_seg for _, _, X_aud_seg in tl[0]] for tl in train_list]
        for img_arr, aud_arr in zip(img_array_train, aud_array_train):
            img_mean_col_train += [np.mean(np.concatenate(img_arr), axis=0, keepdims=True)]
            aud_mean_col_train += [np.mean(np.concatenate(aud_arr), axis=0, keepdims=True)]
            img_std_col_train += [np.std(np.concatenate(img_arr), axis=0, keepdims=True)]
            aud_std_col_train += [np.std(np.concatenate(aud_arr), axis=0, keepdims=True)]
    else:
        img_array_train = np.concatenate([X_img_seg for _, X_img_seg, _ in train_list[0]])
        aud_array_train = np.concatenate([X_aud_seg for _, _, X_aud_seg in train_list[0]])
        img_mean_col_train = np.mean(img_array_train, axis=0, keepdims=True)
        aud_mean_col_train = np.mean(aud_array_train, axis=0, keepdims=True)
        img_std_col_train = np.std(img_array_train, axis=0, keepdims=True)
        aud_std_col_train = np.std(aud_array_train, axis=0, keepdims=True)

    def normalization_one_split(split_list, split_type):

        def normalization_modality(mod_list, mean_col, std_col, mod_type):

            num_all_nan, num_some_nan = 0, 0
            max_percentage_nan = 0
            one_col_with_all_infinite, one_col_with_half_infinite = 0, 0
            num_some_nan_after_imput, max_percentage_nan_col = 0, 0
            new_mod_list = []

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 0

            for i, mod_array in enumerate(mod_list):
                # Calculate percentage of nan/inf values in data_seg
                perc_nan = np.sum(~np.isfinite(mod_array)) / (mod_array.shape[0] * mod_array.shape[1])
                max_percentage_nan = max(perc_nan, max_percentage_nan)
                if perc_nan == 1:
                    num_all_nan += 1
                elif perc_nan > 0:
                    num_some_nan += 1

                # z-normalization per instance and remove nan/infs
                with np.errstate(invalid="ignore"):  # ignore errors which creates nan/infs, will be dealt right after
                    mod_array = (mod_array - mean_col) / (EPS + std_col)

                if np.all(np.logical_or(np.isnan(mod_array), np.isinf(mod_array))):
                    print("Encountered datapoint {} with image data full of NaNs or infs.".format(split_list[2][i]))
                    num_all_nan += 1

                # After imputation, check if there are still infinite values
                num_not_finite = np.sum(~np.isfinite(mod_array))
                if num_not_finite > 0:
                    num_some_nan_after_imput += 1

                # After normalization, non-finite value processing for image and audio
                mod_array, num1, num2, num3 = process_not_finite_values(mod_array, replace_nan)
                one_col_with_all_infinite += num1
                one_col_with_half_infinite += num2
                max_percentage_nan_col = max(num3, max_percentage_nan_col)
                new_mod_list += [mod_array]

                print("[Before normalization, {}]".format(mod_type))
                print("Number of data with all NaN or inf: {}".format(num_all_nan))
                print("Number of data with some NaN or inf: {}".format(num_some_nan))
                print("Max percentage NaN or inf: {}%".format(mod_type, np.round(max_percentage_nan * 100, decimals=2)))
                print("Number of columns with all NaN or inf: {}".format(one_col_with_all_infinite))
                print("Number of columns with at least 40% NaN or inf: {}".format(one_col_with_half_infinite))
                print("Max percentage columns with NaN or inf: {}%".format(np.round(max_percentage_nan_col * 100,
                                                                                    decimals=2)))
                print("\n[After normalization, {}]".format(mod_type))
                print("Number of data with some NaN or inf: {}\n".format(num_some_nan_after_imput))

            return new_mod_list

        if with_cross_val:  # train_list and valid_list contains num_fold elements
            img_norm, aud_norm = [], []
            if split_type == 'valid':
                img_list = [[X_img_seg for _, X_img_seg, _ in split_l[0]] for split_l in split_list]
                aud_list = [[X_aud_seg for _, _, X_aud_seg in split_l[0]] for split_l in split_list]
            elif split_type == 'train':  # 'train' already calculated earlier
                img_list = img_array_train
                aud_list = aud_array_train
            else:  # split_list == 'test'
                img_list = [X_img_seg for _, X_img_seg, _ in split_list[0]]
                aud_list = [X_aud_seg for _, _, X_aud_seg in split_list[0]]
            print("----------------------------------------------------")
            print("Normalization split {}".format(split_type))
            print("----------------------------------------------------")
            for fold_id, (i_mean_col, i_std_col, a_mean_col, a_std_col) in enumerate(zip(img_mean_col_train,
                                                                                         img_std_col_train,
                                                                                         aud_mean_col_train,
                                                                                         aud_std_col_train)):
                i_list = img_list if split_type == 'test' else img_list[fold_id]
                a_list = aud_list if split_type == 'test' else aud_list[fold_id]
                img_norm += [normalization_modality(i_list, i_mean_col, i_std_col, mod_type="image")]
                aud_norm += [normalization_modality(a_list, a_mean_col, a_std_col, mod_type="audio")]

            return [[[img_norm[fold_id], aud_norm[fold_id], split_list[fold_id][0][2]],
                     split_list[fold_id][1], split_list[fold_id][2]] for fold_id in range(len(split_list))]

        else:  # not with_cross_val or (with_cross_val and test)
            img_list = [X_img_seg for _, X_img_seg, _ in split_list[0]] if split_type != 'train' else img_array_train
            aud_list = [X_aud_seg for _, _, X_aud_seg in split_list[0]] if split_type != 'train' else aud_array_train
            print("----------------------------------------------------")
            print("Normalization split {}".format(split_type))
            print("----------------------------------------------------")
            img_norm = normalization_modality(img_list, img_mean_col_train, img_std_col_train, mod_type="image")
            aud_norm = normalization_modality(aud_list, aud_mean_col_train, aud_std_col_train, mod_type="audio")

            return [[img_norm, aud_norm, split_list[0][2]], split_list[1], split_list[2]]

    new_train_list = normalization_one_split(train_list, split_type='train')
    new_valid_list = normalization_one_split(valid_list, split_type='valid')
    new_test_list = normalization_one_split(test_list, split_type='test')

    return new_train_list, new_valid_list, new_test_list


def get_length_dataset(train_list, valid_list, test_list, param_task, is_tf):
    """
    Return the desired length for creating batches. (None if we want to keep variable lengths)
    :param train_list: [X_train, y_train, seg_train]
    :param valid_list: [X_valid, y_valid, seg_valid]
    :param test_list: [X_test, y_test, seg_test]
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
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






