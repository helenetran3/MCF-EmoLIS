import torch
import numpy as np


class DatasetTorch(torch.utils.data.Dataset):

    def __init__(self, x_list, y, seg_list, num_steps, one_hot, device):
        """
        :param x_list: list of lists of 3 arrays of shape (number steps, number features)
        :param y: list (or dict of 2 lists, keys=['coarse_task','fine_task']) of arrays of shape (1, 7), for the 7 emotions
        :param seg_list: list of ids of the segment described by (x, y). Example: 'zk2jTlAtvSU[1]'
        """
        self.x_list = x_list
        self.y = y
        self.seg_list = seg_list
        self.num_steps = num_steps
        self.oh = one_hot
        self.device = device

    @staticmethod
    def seq_fixed_length(seq_array, fixed_num_steps):
        """
        Pad or truncate sequence data to a given length.

        :param seq_array: 1 array of shape (own number of steps, number features)
        :param fixed_num_steps: max number of steps
        :return: seq_array with the fixed number of steps
        """
        own_num_steps = seq_array.shape[0]
        num_features = seq_array.shape[1]

        if fixed_num_steps > own_num_steps:  # padding
            diff = fixed_num_steps - own_num_steps
            zeroes = np.zeros((diff, num_features))
            new_seq_array = np.concatenate((seq_array, zeroes))

        elif fixed_num_steps < own_num_steps:  # truncate
            new_seq_array = seq_array[:fixed_num_steps, :]

        else:  # length seq_array equals to fixed_num_steps
            new_seq_array = seq_array

        return new_seq_array

    def __getitem__(self, index):
        if self.num_steps is not None:
            x_text_i = self.seq_fixed_length(self.x_list[index][0], self.num_steps)
            x_image_i = self.seq_fixed_length(self.x_list[index][1], self.num_steps)
            x_audio_i = self.seq_fixed_length(self.x_list[index][2], self.num_steps)
        else:
            x_text_i = self.x_list[index][0]
            x_image_i = self.x_list[index][1]
            x_audio_i = self.x_list[index][2]

        x_text_i = torch.from_numpy(x_text_i).to(self.device)
        x_image_i = torch.from_numpy(x_image_i).to(self.device)
        x_audio_i = torch.from_numpy(x_audio_i).to(self.device)

        if type(self.y) == dict:
            y_i = {}
            for task_i in ['coarse_task', 'fine_task']:
                y_index = self.y[task_i][index]
                y_index = y_index if self.oh[task_i] else y_index.flatten()
                y_i[task_i] = torch.from_numpy(y_index).to(self.device)
        else:
            y_i = self.y[index]
            y_i = y_i if self.oh else y_i.flatten()  # To deal with one-hot encoded labels
            y_i = torch.from_numpy(y_i).to(self.device)

        return (x_text_i, x_image_i, x_audio_i), y_i  # , seg_i

    def __len__(self):
        return len(self.y_list)


def collate_fn_for_variable_length(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.stack(target, 0)

    return [data, target]
