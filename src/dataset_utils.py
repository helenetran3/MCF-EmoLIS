import re
import numpy as np
from mmsdk import mmdatasdk
from src.pickle_functions import *
from statistics import mean
from sklearn.model_selection import KFold


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
    :param param_task: dictionary of 'task' hyperparameters (cf. param_task.json)
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

    :param param_task: dictionary of 'task' hyperparameters (cf. param_task.json)
    :return: 3 lists of ids for training, validation, and test sets respectively
    """

    with_custom_split = param_task['with_custom_split']

    train_ids = mmdatasdk.cmu_mosei.standard_folds.standard_train_fold
    valid_ids = mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold
    test_ids = mmdatasdk.cmu_mosei.standard_folds.standard_test_fold

    if with_custom_split:
        train_ids, valid_ids, test_ids = perform_custom_split(train_ids, valid_ids, test_ids)

    return train_ids, valid_ids, test_ids


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
    :param param_task: dictionary of 'task' hyperparameters (cf. param_task.json)
    :param param_log: dictionary of 'logistics' hyperparameters (cf. param_logistics.json)
    :return: 3 lists train_list = [x_train, y_train, seg_train], valid_list = [x_valid, y_valid, seg_valid],
     test_list = [x_test, y_test, seg_test]
    """

    image_feature = param_task['image_feature']
    pickle_name_fold = param_log['pickle_name_fold']

    # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
    EPS = 0

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

        # Remove nan values
        image_seg = np.nan_to_num(image_seg)
        audio_seg = np.nan_to_num(audio_seg)
        label_seg = np.nan_to_num(label_seg)

        # z-normalization per instance and remove nan/infs
        image_seg = np.nan_to_num(
            (image_seg - image_seg.mean(0, keepdims=True)) / (EPS + np.std(image_seg, axis=0, keepdims=True)))
        audio_seg = np.nan_to_num(
            (audio_seg - audio_seg.mean(0, keepdims=True)) / (EPS + np.std(audio_seg, axis=0, keepdims=True)))

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
    print("Number of training datapoints: {} ({:.2f}%)".format(train_size, 100 * (train_size / total_data_split)))
    print("Number of validation datapoints: {} ({:.2f}%)".format(valid_size, 100 * (valid_size / total_data_split)))
    print("Number of test datapoints: {} ({:.2f}%)".format(test_size, 100 * (test_size / total_data_split)))

    # Remove sentiment labels
    y_train_emo = [y[:, 1:] for y in y_train]
    y_valid_emo = [y[:, 1:] for y in y_valid]
    y_test_emo = [y[:, 1:] for y in y_test]

    train_res = [x_train, y_train_emo, seg_train]
    valid_res = [x_valid, y_valid_emo, seg_valid]
    test_res = [x_test, y_test_emo, seg_test]

    # Save lists with pickle
    save_with_pickle(train_res, pickle_name_fold + "_train", pickle_folder="raw_folds", root_folder="cmu_mosei")
    save_with_pickle(valid_res, pickle_name_fold + "_valid", pickle_folder="raw_folds", root_folder="cmu_mosei")
    save_with_pickle(test_res, pickle_name_fold + "_test", pickle_folder="raw_folds", root_folder="cmu_mosei")

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


def get_length_dataset(train_list, valid_list, test_list, param_task, is_tf):
    """
    Return the desired length for creating batches. (None if we want to keep variable lengths)
    :param train_list: [X_train, y_train, seg_train]
    :param valid_list: [X_valid, y_valid, seg_valid]
    :param test_list: [X_test, y_test, seg_test]
    :param param_task: dictionary of 'task' hyperparameters (cf. param_task.json)
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
    elif set_length.is_digit() and int(set_length) > 0:
        return int(set_length)
    else:
        raise ValueError("Invalid value for parameter set_length. Please choose value between: 'avg', 'max',"
                         " any strictly positive number, 'none'")






