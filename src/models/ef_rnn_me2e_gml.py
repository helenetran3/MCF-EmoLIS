import torch
from torch import nn
import torch.nn.functional as F


class MModel(nn.Module):
    """
    Build the early fusion version of the lf_rnn_me3e.
    Works only when the number of steps is the same for all datapoints.

    :param model_id: Model id (int)
    :param param_model: dictionary of 'architecture' hyperparameters of the model chosen (cf. param_model.json)
    :param num_features_text: size of text feature vector (it should be the same at each step)
    :param num_features_image: size of image feature vector (it should be the same at each step)
    :param num_features_audio: size of audio feature vector (it should be the same at each step)
    :param num_classes: number of classes to predict
    """
    def __init__(self, model_id, param_model, num_features_text, num_features_image, num_features_audio, num_classes,
                 num_layers=1, dropout=0.1, bi=True):

        super(MModel, self).__init__()

        self.model_name = 'ef_rnn_me2e_gml'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio
        self.mods = 'tav'  # we use the three modalities here

        self.num_features_tav = num_features_text + num_features_image + num_features_audio
        self.num_scores = 4
        self.num_classes = num_classes

        self.concatenate = torch.cat

        self.rnn = nn.LSTM(input_size=self.num_features_tav,
                           hidden_size=self.num_features_tav,
                           num_layers=num_layers,
                           dropout=(dropout if num_layers > 1 else 0),
                           bidirectional=bi)

        linear_in_size = self.num_features_tav if not bi else self.num_features_tav * 2

        self.affine = nn.Sequential(nn.Linear(linear_in_size, linear_in_size // 2),
                                    nn.ReLU(),
                                    nn.Linear(linear_in_size // 2, self.num_classes * self.num_scores))

    def forward(self, x_text, x_image, x_audio):

        x = self.concatenate((x_text, x_image, x_audio), 2)
        x = x.transpose(0, 1)
        _, (x_hn, _) = self.rnn(x)
        x_hn = x_hn.transpose(0, 1).flatten(start_dim=1)
        x_hn = self.affine(x_hn).view(-1, self.num_classes, self.num_scores)
        x_hn = F.softmax(x_hn, dim=2)

        return x_hn
