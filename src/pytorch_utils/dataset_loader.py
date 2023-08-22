import torch


class DatasetTorch(torch.utils.data.Dataset):

    def __init__(self, x_list, y_list, seg_list, device):
        """
        :param x_list: list of lists of 3 arrays of shape (number steps, number features)
        :param y_list: list of arrays of shape (1, 7), for the 7 emotions
        :param seg_list: list of ids of the segment described by (x, y). Example: 'zk2jTlAtvSU[1]'
        """
        self.x_list = x_list
        self.y_list = y_list
        self.seg_list = seg_list
        self.device = device

    def __getitem__(self, index):
        x_text_i = torch.from_numpy(self.x_list[index][0]).to(self.device)
        x_image_i = torch.from_numpy(self.x_list[index][1]).to(self.device)
        x_audio_i = torch.from_numpy(self.x_list[index][2]).to(self.device)
        y_i = torch.from_numpy(self.y_list[index].flatten()).to(self.device)
        return (x_text_i, x_image_i, x_audio_i), y_i  # , seg_i

    def __len__(self):
        return len(self.y_list)


def collate_fn_for_variable_length(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.stack(target, 0)

    return [data, target]
