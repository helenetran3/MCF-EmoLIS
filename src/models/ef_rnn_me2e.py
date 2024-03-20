import torch
from torch import nn


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

        self.model_name = 'ef_rnn_me2e'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio
        self.mods = 'tav'  # we use the three modalities here

        num_features_tav = num_features_text + num_features_image + num_features_audio

        self.concatenate = torch.cat

        self.rnn = nn.LSTM(input_size=num_features_tav,
                           hidden_size=num_features_tav,
                           num_layers=num_layers,
                           dropout=(dropout if num_layers > 1 else 0),
                           bidirectional=bi)

        linear_in_size = num_features_tav if not bi else num_features_tav * 2

        self.affine = nn.Sequential(nn.Linear(linear_in_size, linear_in_size // 2),
                                    nn.ReLU(),
                                    nn.Linear(linear_in_size // 2, num_classes),
                                    nn.Sigmoid())

    def forward(self, x_text, x_image, x_audio):

        x = self.concatenate((x_text, x_image, x_audio), 2)
        x = x.transpose(0, 1)
        _, (x_hn, _) = self.rnn(x)
        x_hn = x_hn.transpose(0, 1).flatten(start_dim=1)
        x_hn = self.affine(x_hn)

        return x_hn

        # all_logits = []
        #
        # if 't' in self.mods:
        #     output_t, _ = self.rnns['t'](x_text.transpose(0, 1))
        #     output_t = output_t[-1, :, :]
        #     output_t = self.affines['t'](output_t)
        #     all_logits.append(output_t)
        #
        # if 'a' in self.mods:
        #     # max_len = max(audio_features_lens)
        #     # audio_features = audio_features.split(audio_features_lens, dim=0)
        #     # audio_features = [padTensor(s, max_len) for s in audio_features]
        #     # audio_features = torch.stack(audio_features, dim=1)  # (seq_len, batch, dim)
        #     x_audio = x_audio.transpose(0, 1)
        #     _, (audio_hn, _) = self.rnns['a'](x_audio)
        #     audio_hn = audio_hn.transpose(0, 1).flatten(start_dim=1)  # (batch, hid_dim * 2)
        #     audio_hn = self.affines['a'](audio_hn)
        #     all_logits.append(audio_hn)
        #
        # if 'v' in self.mods:
        #     # max_len = max(img_features_lens)
        #     # img_features = img_features.split(img_features_lens, dim=0)
        #     # img_features = [padTensor(s, max_len) for s in img_features]
        #     # img_features = torch.stack(img_features, dim=1)  # (seq_len, batch, dim)
        #     x_image = x_image.transpose(0, 1)
        #     _, (img_hn, _) = self.rnns['v'](x_image)
        #     img_hn = img_hn.transpose(0, 1).flatten(start_dim=1)  # (batch, hid_dim * 2)
        #     img_hn = self.affines['v'](img_hn)
        #     all_logits.append(img_hn)
        #
        # if len(self.mods) == 1:
        #     return all_logits[0]
        #
        # return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)
