import numpy as np
import torch
from torch import nn
from src.models.me2e.modules.transformer_encoder import WrappedTransformerEncoder

from src.label_processing import create_label_multitask


class MModel(nn.Module):
    def __init__(self, model_id, param_model, param_task, num_features_text, num_features_image, num_features_audio,
                 num_classes):
        super(MModel, self).__init__()

        self.model_name = 'lf_rnn_me2e'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio
        self.num_classes = num_classes
        self.num_scores = 4
        self.mods = 'tav'  # we use the three modalities here
        label_type = param_task['label_type']
        self.label_type, _ = create_label_multitask(label_type) if type(label_type) == list else (label_type, None)
        self.multitask = type(self.label_type) == dict

        nlayers = param_model['trans_nlayers']
        num_features_tav_incr = num_features_text + num_features_image + num_features_audio + 1 # 409+1 for div by 5
        compute_dim = lambda l: self.num_classes * self.num_scores if 'oh' in l else self.num_classes
        output_dim = [compute_dim(self.label_type['coarse_task']), compute_dim(self.label_type['fine_task'])] \
            if self.multitask else [compute_dim(self.label_type)]

        # Multitask parameters
        num_branches = 2 if self.multitask else 1
        proj_size_mt = param_model['mt_proj_size']  # set to 60 for a multiple of 5
        num_features_for_trans_and_aff = [num_features_tav_incr, proj_size_mt]

        self.transformer = nn.ModuleList([WrappedTransformerEncoder(dim=num_features_for_trans_and_aff[i],
                                                                    num_layers=nlayers,
                                                                    num_heads=5) for i in range(num_branches)])

        self.affine = nn.ModuleList([nn.Sequential(nn.Linear(num_features_for_trans_and_aff[i],
                                                             num_features_for_trans_and_aff[i] // 2),
                                                   nn.ReLU(),
                                                   nn.Linear(num_features_for_trans_and_aff[i] // 2, output_dim[i]))
                                     for i in range(num_branches)])

        if self.multitask:
            self.proj = nn.Conv1d(2*num_features_tav_incr, proj_size_mt, kernel_size=1, padding=0, bias=False)

    def final_activation(self, output, label_i):
        if label_i in ['present', 'dominant']:
            return torch.sigmoid(output)
        elif 'oh' in label_i:
            return torch.softmax(output.view(-1, self.num_classes, self.num_scores), dim=2)
        else:
            return torch.relu(output)

    def forward(self, x_text, x_image, x_audio):

        max_all_lens = x_text.size(1)  # is set_length parameter in config/param_training_task.json
        assert max_all_lens == x_image.size(1) == x_audio.size(1)
        num_samples = x_text.size(0)
        assert num_samples == x_image.size(0) == x_audio.size(0)

        zero_vector = torch.zeros(num_samples, max_all_lens, 1)  # for division by 5 in transformer for num_features
        x = torch.cat((x_text, x_image, x_audio, zero_vector), 2) # (num_samples, t, num_features)
        all_lens = self.get_len_samples(x)  # list of sequence lengths in the training batch
        x_cls, x_trans_main = self.transformer[0](x, all_lens, max_all_lens, get_cls=True) # (num_samples, num_features)
        output_main = self.affine[0](x_cls)

        if self.multitask:
            x_multi = torch.cat([x, x_trans_main], dim=2)  # (num_samples, t, num_features*2)
            x_multi = x_multi.permute(0, 2, 1)  # (num_samples, num_features*2, t)
            x_multi = self.proj(x_multi)  # (num_samples, proj_size, t)
            x_multi = x_multi.permute(0, 2, 1)  # (num_samples, t, proj_size)
            x_cls_multi, _ = self.transformer[1](x_multi, all_lens, max_all_lens, get_cls=True)
            output_multi = self.affine[1](x_cls_multi)
            output = {"coarse_task": self.final_activation(output_main, self.label_type['coarse_task']),
                      "fine_task": self.final_activation(output_multi, self.label_type['fine_task'])}
        else:
            output = self.final_activation(output_main, self.label_type)

        return output

    @staticmethod
    def get_len_samples(x_feat):
        """
        Return a list of the sequence length of each sample from x_feat, based on the last index with non-zero value.

        x_feat: array of shape (num_samples, num_features) where padding has already been performed.
        mod_name: name of the modality ('image' or 'audio')
        """

        def squeeze_feat_dim(x_array):
            x_sq = []
            for ind in range(num_samples):
                x_sq += [[int(any(x_array[ind, i, :])) for i in range(seq_len)]]  # (seq_len,)
            x_sq = np.array(x_sq)
            return x_sq

        def get_len_one_sample(x_feat_i, i):
            x_nonzero = np.nonzero(x_feat_i)
            real_len = 0 if x_nonzero[0].shape[0] == 0 else np.max(x_nonzero) + 1  # +1 because we want the length
            # if real_len == 0:
            #     print("Note from mme2e.py: {}-th feature is a zero vector.".format(i))
            return real_len

        num_samples, seq_len = x_feat.size(0), x_feat.size(1)
        x = x_feat.numpy()  # (num_samples, seq, feat)
        x_squeezed = squeeze_feat_dim(x)  # (num_samples, seq)
        lens = [get_len_one_sample(x_squeezed[i, :], i) for i in range(num_samples)]
        return lens
