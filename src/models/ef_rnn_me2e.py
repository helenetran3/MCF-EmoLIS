import torch
from torch import nn

from src.label_processing import create_label_multitask

class MModel(nn.Module):
    """
    Build the early fusion version of the lf_rnn_me3e.
    Works only when the number of steps is the same for all datapoints.

    :param model_id: Model id (int)
    :param param_model: dictionary of 'architecture' hyperparameters of the model chosen (cf. param_model.json)
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param num_features_text: size of text feature vector (it should be the same at each step)
    :param num_features_image: size of image feature vector (it should be the same at each step)
    :param num_features_audio: size of audio feature vector (it should be the same at each step)
    :param num_classes: number of classes to predict
    """

    def __init__(self, model_id, param_model, param_task, num_features_text, num_features_image, num_features_audio,
                 num_classes, num_layers=1, dropout=0.1, bi=True):

        super(MModel, self).__init__()

        self.model_name = 'ef_rnn_me2e'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio

        label_type = param_task['label_type']
        self.label_type, _ = create_label_multitask(label_type) if type(label_type) == list else (label_type, None)
        self.multitask = type(self.label_type) == dict

        num_branches = 2 if self.multitask else 1

        self.num_features_total = num_features_text + num_features_image + num_features_audio
        self.num_scores = 4
        self.num_classes = num_classes
        compute_dim = lambda l: self.num_classes * self.num_scores if 'oh' in l else self.num_classes
        output_dim = [compute_dim(self.label_type['coarse_task']), compute_dim(self.label_type['fine_task'])] \
            if self.multitask else [compute_dim(self.label_type)]

        self.rnn_main = nn.LSTM(input_size=self.num_features_total,
                                hidden_size=self.num_features_total,
                                num_layers=num_layers,
                                dropout=(dropout if num_layers > 1 else 0),
                                bidirectional=bi)

        linear_in_size = self.num_features_total if not bi else self.num_features_total * 2
        self.affine_main = nn.Sequential(nn.Linear(linear_in_size, linear_in_size // 2),
                                         nn.ReLU(),
                                         nn.Linear(linear_in_size // 2, output_dim[0]))

        if self.multitask:
            # in the following block, if 0, take the total number of input features (text+audio+image) else take the one set by the user
            proj_size_mt = param_model['mt_proj_size']
            proj_size_total = self.num_features_total if proj_size_mt == 0 else proj_size_mt

            self.proj = nn.Conv1d(3 * self.num_features_total, proj_size_total, kernel_size=1, padding=0, bias=False)

            self.rnn_multi = nn.LSTM(input_size=proj_size_total,
                                     hidden_size=proj_size_total,
                                     num_layers=num_layers,
                                     dropout=(dropout if num_layers > 1 else 0),
                                     bidirectional=bi)

            self.affine_multi = nn.Sequential(nn.Linear(2 * proj_size_total, proj_size_total),
                                              nn.ReLU(),
                                              nn.Linear(proj_size_total, output_dim[1]))




    def final_activation(self, output, label_i):
        if label_i in ['present', 'dominant', 'coarse_prob']:
            return torch.sigmoid(output)
        elif 'oh' in label_i:
            return torch.softmax(output.view(-1, self.num_classes, self.num_scores), dim=2)
        else:
            return torch.relu(output)

    def forward(self, x_text, x_image, x_audio):

        x = torch.cat([x_text, x_image, x_audio], dim=2)  # (num_samples, t, num_features)
        x = x.transpose(0, 1)  # (t, num_samples, num_features)
        # below, out_rnn of size (t, num_samples, 2*num_features)
        #        x_hnn of size (2, num_samples, num_features), 2 for bidirectional
        out_rnn, (x_hn_main, _) = self.rnn_main(x)
        x_hn_main = x_hn_main.transpose(0, 1).flatten(start_dim=1)  # (num_samples, 2 * num_features)
        output_main = self.affine_main(x_hn_main)  # (num_samples, num_classes)

        if self.multitask:
            x_multi = torch.cat([x, out_rnn], dim=2)  # (t, num_samples, 3*num_features)
            x_multi = x_multi.permute(1, 2, 0)  # (num_samples, 3*num_features, t)
            x_multi = self.proj(x_multi)  # (num_samples, proj_size_total, t)
            x_multi = x_multi.permute(2, 0, 1)  # (t, num_samples, proj_size_total)
            _, (x_hn_multi, _) = self.rnn_multi(x_multi)
            x_hn_multi = x_hn_multi.transpose(0, 1).flatten(start_dim=1)
            output_multi = self.affine_multi(x_hn_multi)
            output = {"coarse_task": self.final_activation(output_main, self.label_type['coarse_task']),
                      "fine_task": self.final_activation(output_multi, self.label_type['fine_task'])}
        else:
            output = self.final_activation(output_main, self.label_type)

        return output
