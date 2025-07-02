import numpy as np
import torch
from torch import nn

from src.label_processing import create_label_multitask

class MModel(nn.Module):
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
    def __init__(self, model_id, param_model, param_task, num_features_text, num_features_image, num_features_audio,
                 num_classes, num_layers=1, dropout=0.1, bi=True):

        super(MModel, self).__init__()

        self.model_name = 'lf_rnn_me2e'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio
        self.num_classes = num_classes
        self.num_scores = 4

        # Label type
        label_type = param_task['label_type']
        self.label_type, _ = create_label_multitask(label_type) if type(label_type) == list else (label_type, None)
        self.multitask = type(self.label_type) == dict

        # Compute dimensions
        feature_sizes = np.array([num_features_text, num_features_audio, num_features_image])
        linear_in_sizes = feature_sizes if not bi else feature_sizes * 2
        compute_dim = lambda l: self.num_classes * self.num_scores if 'oh' in l else self.num_classes
        output_dim = [compute_dim(self.label_type['coarse_task']), compute_dim(self.label_type['fine_task'])] \
            if self.multitask else [compute_dim(self.label_type)]

        self.rnns = nn.ModuleDict({
            't': nn.LSTM(
                input_size=num_features_text,
                hidden_size=num_features_text,
                num_layers=num_layers,
                dropout=(dropout if num_layers > 1 else 0),
                bidirectional=bi
            ),
            'a': nn.LSTM(
                input_size=num_features_audio,
                hidden_size=num_features_audio,
                num_layers=num_layers,
                dropout=(dropout if num_layers > 1 else 0),
                bidirectional=bi
            ),
            'v': nn.LSTM(
                input_size=num_features_image,
                hidden_size=num_features_image,
                num_layers=num_layers,
                dropout=(dropout if num_layers > 1 else 0),
                bidirectional=bi
            )
        })

        self.affines = nn.ModuleDict({
            't': nn.Sequential(
                nn.Linear(linear_in_sizes[0], linear_in_sizes[0] // 2),
                nn.ReLU(),
                nn.Linear(linear_in_sizes[0] // 2, output_dim[0])
            ),
            'a': nn.Sequential(
                nn.Linear(linear_in_sizes[1], linear_in_sizes[1] // 2),
                nn.ReLU(),
                nn.Linear(linear_in_sizes[1] // 2, output_dim[0])
            ),
            'v': nn.Sequential(
                nn.Linear(linear_in_sizes[2], linear_in_sizes[2] // 2),
                nn.ReLU(),
                nn.Linear(linear_in_sizes[2] // 2, output_dim[0])
            )
        })

        if self.multitask:
            # in the following block, if 0, take the total number of input features (text+audio+image) else take the one set by the user
            proj_size_mt_text = param_model['mt_proj_size_text']
            proj_size_mt_audio = param_model['mt_proj_size_audio']
            proj_size_mt_image = param_model['mt_proj_size_image']
            proj_size_mt_text = num_features_text if proj_size_mt_text == 0 else proj_size_mt_text
            proj_size_mt_audio = num_features_audio if proj_size_mt_audio == 0 else proj_size_mt_audio
            proj_size_mt_image = num_features_image if proj_size_mt_image == 0 else proj_size_mt_image
            proj_size_total = proj_size_mt_text + proj_size_mt_audio + proj_size_mt_image

            self.proj_t = nn.Conv1d(3 * num_features_text, proj_size_mt_text, kernel_size=1, padding=0, bias=False)
            self.proj_a = nn.Conv1d(3 * num_features_audio, proj_size_mt_audio, kernel_size=1, padding=0, bias=False)
            self.proj_v = nn.Conv1d(3 * num_features_image, proj_size_mt_image, kernel_size=1, padding=0, bias=False)
            self.rnn_multi = nn.LSTM(input_size=proj_size_total,
                                     hidden_size=proj_size_total,
                                     num_layers=num_layers,
                                     dropout=(dropout if num_layers > 1 else 0),
                                     bidirectional=bi)
            self.affine_multi = nn.Sequential(
                nn.Linear(2 * proj_size_total, proj_size_total),
                nn.ReLU(),
                nn.Linear(proj_size_total, output_dim[1]))

        self.weighted_fusion = nn.Linear(3, 1, bias=False)  # 3 modalities to fuse into 1

    def final_activation(self, output, label_i):
        if label_i in ['present', 'dominant']:
            return torch.sigmoid(output)
        elif 'oh' in label_i:
            return torch.softmax(output.view(-1, self.num_classes, self.num_scores), dim=2)
        else:
            return torch.relu(output)

    def forward(self, x_text, x_image, x_audio):

        all_logits = []

        # Apply RNN to text modality
        output_t, _ = self.rnns['t'](x_text.transpose(0, 1))
        output_t_logits = output_t[-1, :, :]
        output_t_logits = self.affines['t'](output_t_logits)
        all_logits.append(output_t_logits)

        # Apply RNN to audio modality
        output_a, (audio_hn, _) = self.rnns['a'](x_audio.transpose(0, 1))
        audio_hn = audio_hn.transpose(0, 1).flatten(start_dim=1)  # (batch, hid_dim * 2)
        audio_hn = self.affines['a'](audio_hn)
        all_logits.append(audio_hn)

        # Apply RNN to image modality
        output_v, (img_hn, _) = self.rnns['v'](x_image.transpose(0, 1))
        img_hn = img_hn.transpose(0, 1).flatten(start_dim=1)  # (batch, hid_dim * 2)
        img_hn = self.affines['v'](img_hn)
        all_logits.append(img_hn)

        # Weighted fusion
        output_main = self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)

        if self.multitask:
            output_t = output_t.transpose(0, 1)
            output_a = output_a.transpose(0, 1)
            output_v = output_v.transpose(0, 1)
            x_multi_t = torch.cat([x_text, output_t], dim=2)  # (num_samples, t, 3*num_features_text)
            x_multi_a = torch.cat([x_audio, output_a], dim=2)  # (num_samples, t, 3*num_features_audio)
            x_multi_v = torch.cat([x_image, output_v], dim=2)  # (num_samples, t, 3*num_features_image)
            x_multi_t = x_multi_t.transpose(1, 2)  # (num_samples, num_features_text*3, t)
            x_multi_a = x_multi_a.transpose(1, 2)  # (num_samples, num_features_audio*3, t)
            x_multi_v = x_multi_v.transpose(1, 2)  # (num_samples, num_features_mage*3, t)
            x_multi_t = self.proj_t(x_multi_t)  # (num_samples, proj_size_mt_text, t)
            x_multi_a = self.proj_a(x_multi_a)  # (num_samples, proj_size_mt_audio, t)
            x_multi_v = self.proj_v(x_multi_v)  # (num_samples, proj_size_mt_mage, t)
            x_multi_t = x_multi_t.transpose(1, 2)  # (num_samples, t, proj_size_mt_text)
            x_multi_a = x_multi_a.transpose(1, 2)  # (num_samples, t, proj_size_mt_audio)
            x_multi_v = x_multi_v.transpose(1, 2)  # (num_samples, t, proj_size_mt_image)
            x_multi = torch.cat([x_multi_t, x_multi_a, x_multi_v], dim=2) # (num_samples, t, proj_size_total)
            _, (x_hn_multi, _) = self.rnn_multi(x_multi.transpose(0, 1)) # (2, num_samples, proj_size_total)
            x_hn_multi = x_hn_multi.transpose(0, 1).flatten(start_dim=1) # (num_samples, 2*proj_size_total)
            output_multi = self.affine_multi(x_hn_multi) # (num_samples, num_output)
            output = {"coarse_task": self.final_activation(output_main, self.label_type['coarse_task']),
                      "fine_task": self.final_activation(output_multi, self.label_type['fine_task'])}

        else:
            output = self.final_activation(output_main, self.label_type)

        return output