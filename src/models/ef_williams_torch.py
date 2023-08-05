import torch
from torch import nn


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
        self.model_name = 'ef_williams'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio
        self.num_features = num_features_text + num_features_image + num_features_audio
        self.num_classes = num_classes
        self.num_layers = param_archi['num_layers']
        self.num_nodes = param_archi['num_nodes']
        self.dropout_rate = param_archi['dropout_rate']
        self.final_activ = param_archi['final_activ']
        self.concatenate = torch.cat
        self.batch_norm = nn.BatchNorm1d(self.num_features)
        self.bidir_lstm_layers = nn.LSTM(self.num_features, 64, self.num_layers, bidirectional=True).to(device="cpu", dtype=torch.bfloat16)
        self.dropout1 = nn.Dropout(self.dropout_rate).to(device='cpu', dtype=torch.bfloat16)
        self.dropout2 = nn.Dropout(self.dropout_rate).to(device='cpu', dtype=torch.bfloat16)
        self.linear1 = nn.Linear(2*64, self.num_nodes).to(device='cpu', dtype=torch.bfloat16)
        self.linear2 = nn.Linear(self.num_nodes, self.num_classes).to(device='cpu', dtype=torch.bfloat16)

    def print_values(self):
        print("Model name:", self.model_name)
        print("Model id:", self.model_id)
        print("Number layers:", self.num_layers)
        print("Number nodes for the penultimate dense layer:", self.num_nodes)
        print("Dropout rate:", self.dropout_rate)
        print("Final activation:", self.final_activ)

    def forward(self, x_text, x_image, x_audio):
        # print("x_text.size():", x_text.size())
        # print("x_image.size():", x_image.size())
        # print("x_audio.size():", x_audio.size())
        x = self.concatenate((x_text, x_image, x_audio), 1)
        bnorm = self.batch_norm(x)
        blstm = self.bidir_lstm_layers(bnorm)[0][-1]  # takes the last output (-1) in the output sequence (0)
        blstm = self.dropout1(blstm)
        l1 = self.linear1(blstm)
        l1 = self.dropout2(l1)
        out = self.linear2(l1)
        return out


def get_data_for_csv(model_name, param_archi):
    """
    Get header and data to save model hyperparameters in csv file (ie. number of layers, nodes, dropout rate).

    :param model_name: Model name (string)
    :param param_archi: dictionary of 'architecture' hyperparameters of the model chosen (cf. hyperparameters.json)
    :return: two lists of strings (one for header, one for data) to save in csv
    """

    data = [model_name, param_archi['num_layers'], param_archi['num_nodes'], param_archi['dropout_rate'],
            param_archi['final_activ']]
    header = ['model_name', 'num_layers', 'num_nodes', 'dropout_rate', 'final_activ']

    return header, data
