import torch
from torch import nn


class MModel(nn.Module):
    def __init__(self, model_id, param_model, num_features_text, num_features_image, num_features_audio, num_classes):
        """
        Build the model described in the paper (cf. README for the reference).
        Works only when the number of steps is the same for all datapoints.

        :param model_id: Model id (int)
        :param param_model: dictionary of 'architecture' hyperparameters of the model chosen (cf. param_model.json)
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
        self.num_layers = param_model['num_layers']
        self.num_nodes = param_model['num_nodes']
        self.dropout_rate = param_model['dropout_rate']
        self.final_activ = param_model['final_activ']
        self.concatenate = torch.cat
        self.batch_norm = nn.BatchNorm1d(self.num_features)
        self.bidir_lstm_layers = []
        self.dropout_layers = []
        for i in range(self.num_layers):
            input_bidir_layers = self.num_features if i == 0 else 128
            self.bidir_lstm_layers += [nn.LSTM(input_bidir_layers, 64, self.num_layers, bidirectional=True)
                                           .to(device="cpu", dtype=torch.bfloat16)]
            self.dropout_layers += [nn.Dropout(self.dropout_rate).to(device='cpu', dtype=torch.bfloat16)]
        self.dropout_end = nn.Dropout(self.dropout_rate).to(device='cpu', dtype=torch.bfloat16)
        self.linear1 = nn.Linear(2*64, self.num_nodes).to(device='cpu', dtype=torch.bfloat16)
        self.linear2 = nn.Linear(self.num_nodes, self.num_classes).to(device='cpu', dtype=torch.bfloat16)

    def forward(self, x_text, x_image, x_audio):
        # print("x_text.size():", x_text.size())
        # print("x_image.size():", x_image.size())
        # print("x_audio.size():", x_audio.size())
        blstm = None
        x = self.concatenate((x_text, x_image, x_audio), 1)
        bnorm = self.batch_norm(x)
        all_layers = range(self.num_layers)
        for i in all_layers:
            if i == 0:
                blstm = self.bidir_lstm_layers[i](bnorm)[0]  # first element = output sequence
            elif i != self.num_layers-1:
                blstm = self.bidir_lstm_layers[i](blstm)[0]
            else:
                blstm = self.bidir_lstm_layers[i](blstm)[0]
            if i == self.num_layers-1:
                blstm = blstm[-1] # takes the last output (-1)
            blstm = self.dropout_layers[i](blstm)
        l1 = self.linear1(blstm)
        l1 = self.dropout_end(l1)
        out = self.linear2(l1)
        return out
