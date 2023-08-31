import torch
from torch import nn
import torch.nn.functional as F


class MModel(nn.Module):
    def __init__(self, model_id, param_archi, num_features_text, num_features_image, num_features_audio, num_classes):
        """
        Build the model described in the paper (cf. README for the reference).
        Works only when the number of steps is the same for all datapoints.

        :param model_id: Model id (int)
        :param param_archi: dictionary of 'architecture' hyperparameters of the model chosen (cf. hyperparameters.json)
        :param num_features_text: size of text feature vector (it should be the same at each step)
        :param num_features_image: size of image feature vector (it should be the same at each step)
        :param num_features_audio: size of audio feature vector (it should be the same at each step)
        :param num_classes: number of classes to predict
        """
        super().__init__()
        self.model_name = 'ef_from_mfn'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio
        self.num_features = num_features_text + num_features_image + num_features_audio
        self.num_classes = num_classes
        self.num_layers = param_archi['num_layers']
        self.dropout_rate = param_archi['dropout_rate']
        self.h = param_archi['h']
        self.concatenate = torch.cat
        self.bidir_lstm_layers = nn.LSTM(self.num_features, 64, self.num_layers).to(device="cpu", dtype=torch.bfloat16)
        self.lstm = nn.LSTMCell(self.num_features, self.h).to(device='cpu', dtype=torch.bfloat16)
        self.linear1 = nn.Linear(self.h, self.h).to(device='cpu', dtype=torch.bfloat16)
        self.linear2 = nn.Linear(self.h, num_classes).to(device='cpu', dtype=torch.bfloat16)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x_text, x_image, x_audio):
        # print("x_text.size():", x_text.size())
        # print("x_image.size():", x_image.size())
        # print("x_audio.size():", x_audio.size())

        x = self.concatenate((x_text, x_image, x_audio), 1)
        last_hs = self.lstm(x)[1][-1]
        l1 = F.relu(self.linear1(last_hs))
        l1 = self.dropout(l1)
        out = self.linear2(l1)
        return out

        # x = self.concatenate((x_text, x_image, x_audio), 1)
        # bnorm = self.batch_norm(x)
        # blstm = self.bidir_lstm_layers(bnorm)[0][-1]  # takes the last output (-1) in the output sequence (0)
        # blstm = self.dropout1(blstm)
        # l1 = self.linear1(blstm)
        # l1 = self.dropout2(l1)
        # out = self.linear2(l1)
        # return out


def get_data_for_csv(model_name, param_archi, num_classes):
    """
    Get header and data to save model hyperparameters in csv file (ie. number of layers, nodes, dropout rate).

    :param model_name: Model name (string)
    :param param_archi: dictionary of 'architecture' hyperparameters of the model chosen (cf. hyperparameters.json)
    :param num_classes: number of emotion classes (7 with neutral class, else 6)
    :return: two lists of strings (one for header, one for data) to save in csv
    """

    data = [model_name, param_archi['num_layers'], param_archi['h'], param_archi['dropout_rate'], num_classes]
    header = ['model_name', 'num_layers', 'h', 'dropout_rate', 'num_classes']

    return header, data
