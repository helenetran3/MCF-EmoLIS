import tensorflow as tf
import numpy as np


# This file comprises data loader functions in TensorFlow and PyTorch that will be used for model training

# BUILD DATA GENERATOR AND TENSORFLOW DATASETS

class Dataset_tf:

    def __init__(self, x_list, y_list, seg_list, num_steps, num_classes, batch_size, train_mode):
        """
        :param x_list: list of lists of 3 arrays of shape (number steps, number features)
        :param y_list: list of arrays of shape (1, 7), for the 7 emotions
        :param seg_list: list of ids of the segment described by (x, y). Example: 'zk2jTlAtvSU[1]'
        :param num_steps: total number of timesteps
        :param num_classes: number of classes to predict
        :param batch_size: batch size
        :param train_mode: whether we use the dataset for training or not (if not, it is for testing)
        """
        self.x_list = x_list
        self.y_list = y_list
        self.seg_list = seg_list
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.train_mode = train_mode

    @staticmethod
    def seq_padding(seq_array, max_num_steps):
        """
        Pad or truncate sequence data to a given length.

        :param seq_array: 1 array of shape (own number of steps, number features)
        :param max_num_steps: max number of steps
        :return: seq_array with the fixed number of steps
        """
        own_num_steps = seq_array.shape[0]
        num_features = seq_array.shape[1]

        if max_num_steps >= own_num_steps:
            diff = max_num_steps - own_num_steps
            zeroes = np.zeros((diff, num_features))
            new_seq_array = np.concatenate((seq_array, zeroes))

        else:
            raise ValueError("The length of the current feature array should not exceed the maximum number of steps.")

        return new_seq_array

    def datapoint_generator(self):
        """
        Yields successively one datapoint represented by 1 array of features, 1 array of labels, 1 list of start/end times

        :return: yields 1 datapoint (x, y, seg_id) iteratively
        """

        if not len(self.x_list) == len(self.y_list) == len(self.seg_list):
            print("x_list, y_list, seg_list do not have the same number of elements")

        else:
            for x_list_i, y_list_i, seg_list_i in zip(self.x_list, self.y_list, self.seg_list):  # loop on each sample
                x_list_i_0 = self.seq_padding(x_list_i[0], self.num_steps)  # Pad sequence data
                x_list_i_1 = self.seq_padding(x_list_i[1], self.num_steps)
                x_list_i_2 = self.seq_padding(x_list_i[2], self.num_steps)
                yield {"text": x_list_i_0, "image": x_list_i_1, "audio": x_list_i_2}, y_list_i  # , seg_list_i

    def build(self):
        """
        Returns a TensorFlow dataset from a datapoint generator.

        :return: a TensorFlow dataset
        """

        num_feat_text = self.x_list[0][0].shape[1]  # Here we consider that all samples have the same number of features
        num_feat_image = self.x_list[0][1].shape[1]
        num_feat_audio = self.x_list[0][2].shape[1]

        tf_dataset = tf.data.Dataset.from_generator(generator=lambda: self.datapoint_generator(),
                                                    output_signature=(
                                                        {"text": tf.TensorSpec(shape=(self.num_steps, num_feat_text),
                                                                               dtype=tf.float64),
                                                         "image": tf.TensorSpec(shape=(self.num_steps, num_feat_image),
                                                                                dtype=tf.float64),
                                                         "audio": tf.TensorSpec(shape=(self.num_steps, num_feat_audio),
                                                                                dtype=tf.float64)},
                                                        tf.TensorSpec(shape=(1, self.num_classes), dtype=tf.float64)
                                                        # tf.TensorSpec(shape=(), dtype=tf.string)
                                                    ))
        tf_dataset = tf_dataset.shuffle(len(self.seg_list)).batch(self.batch_size).repeat() if self.train_mode \
            else tf_dataset.batch(1).repeat(1)

        return tf_dataset
