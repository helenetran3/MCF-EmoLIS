import numpy as np
import torch
from torch import nn
from typing import List
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

        # Label type
        label_type = param_task['label_type']
        self.label_type, _ = create_label_multitask(label_type) if type(label_type) == list else (label_type, None)
        self.multitask = type(self.label_type) == dict

        # Compute output dimensions
        compute_dim = lambda l: self.num_classes * self.num_scores if 'oh' in l else self.num_classes
        output_dim = [compute_dim(self.label_type['coarse_task']), compute_dim(self.label_type['fine_task'])] \
            if self.multitask else [compute_dim(self.label_type)]

        nlayers = param_model['trans_nlayers']
        # nheads = args['trans_nheads']
        # trans_dim = args['trans_dim']

        self.transformers = nn.ModuleDict({
            't': WrappedTransformerEncoder(
                dim=num_features_text, # 300
                num_layers=nlayers,
                num_heads=4
            ),
            'a': WrappedTransformerEncoder(
                dim=num_features_audio, # 2 empty features are added to make it 144, easy to be divided by #heads
                num_layers=nlayers,
                num_heads=2
            ),
            'v': WrappedTransformerEncoder(
                dim=num_features_image, # 35
                num_layers=nlayers,
                num_heads=5
            )
        })

        self.affines_main = nn.ModuleDict({
            't': nn.Sequential(
                nn.Linear(num_features_text, num_features_text // 2),
                nn.ReLU(),
                nn.Linear(num_features_text // 2, output_dim[0])
            ),
            'a': nn.Sequential(
                nn.Linear(num_features_audio, num_features_audio // 2),
                nn.ReLU(),
                nn.Linear(num_features_audio // 2, output_dim[0])
            ),
            'v': nn.Sequential(
                nn.Linear(num_features_image, num_features_image // 2),
                nn.ReLU(),
                nn.Linear(num_features_image // 2, output_dim[0])
            )
        })

        if self.multitask:
            num_layers_rnn = 1
            dropout = 0.1
            bi = True

            # in the following block, if 0, take the total number of input features (text+audio+image) else take the one set by the user
            proj_size_mt_text = param_model['mt_proj_size_text']
            proj_size_mt_audio = param_model['mt_proj_size_audio']
            proj_size_mt_image = param_model['mt_proj_size_image']
            proj_size_mt_text = num_features_text if proj_size_mt_text == 0 else proj_size_mt_text
            proj_size_mt_audio = num_features_audio if proj_size_mt_audio == 0 else proj_size_mt_audio
            proj_size_mt_image = num_features_image if proj_size_mt_image == 0 else proj_size_mt_image
            proj_size_total = proj_size_mt_text + proj_size_mt_audio + proj_size_mt_image

            self.proj_t = nn.Conv1d(2 * num_features_text, proj_size_mt_text, kernel_size=1, padding=0, bias=False)
            self.proj_a = nn.Conv1d(2 * num_features_audio, proj_size_mt_audio, kernel_size=1, padding=0, bias=False)
            self.proj_v = nn.Conv1d(2 * num_features_image, proj_size_mt_image, kernel_size=1, padding=0, bias=False)

            # self.transformer_multi = WrappedTransformerEncoder(
            #     dim=proj_size_total,
            #     num_layers=nlayers,
            #     num_heads=4)
            self.rnn_multi = nn.LSTM(input_size=proj_size_total,
                                     hidden_size=proj_size_total,
                                     num_layers=num_layers_rnn,
                                     dropout=(dropout if num_layers_rnn > 1 else 0),
                                     bidirectional=bi)

            self.affine_multi = nn.Sequential(
                nn.Linear(2*proj_size_total, proj_size_total),
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

        max_all_lens = x_text.size(1)
        assert max_all_lens == x_image.size(1) == x_audio.size(1)

        text_features_lens = self.get_len_samples(x_text, 'text')
        audio_features_lens = self.get_len_samples(x_audio, 'audio')
        img_features_lens = self.get_len_samples(x_image, 'image')

        # Apply transformer to text modality
        texts_cls, x_trans_main_t = self.transformers['t'](x_text, text_features_lens, max_all_lens, get_cls=True)
        texts = self.affines_main['t'](texts_cls)
        all_logits.append(texts)

        # Apply transformer to acoustic modality
        audio_features_cls, x_trans_main_a = self.transformers['a'](x_audio, audio_features_lens, max_all_lens, get_cls=True)
        audio_features = self.affines_main['a'](audio_features_cls)
        all_logits.append(audio_features)

        # Apply transformer to visual modality
        img_features_cls, x_trans_main_v = self.transformers['v'](x_image, img_features_lens, max_all_lens, get_cls=True)
        img_features = self.affines_main['v'](img_features_cls)
        all_logits.append(img_features)

        output_main = self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)

        if self.multitask:
            x_multi_t = torch.cat([x_text, x_trans_main_t], dim=2)  # (num_samples, t, num_features_text*2)
            x_multi_a = torch.cat([x_audio, x_trans_main_a], dim=2)  # (num_samples, t, num_features_audio*2)
            x_multi_v = torch.cat([x_image, x_trans_main_v], dim=2)  # (num_samples, t, num_features_image*2)
            x_multi_t = x_multi_t.permute(0, 2, 1)  # (num_samples, num_features_text*2, t)
            x_multi_a = x_multi_a.permute(0, 2, 1)  # (num_samples, num_features_audio*2, t)
            x_multi_v = x_multi_v.permute(0, 2, 1)  # (num_samples, num_features_mage*2, t)
            x_multi_t = self.proj_t(x_multi_t)  # (num_samples, proj_size_mt_text, t)
            x_multi_a = self.proj_a(x_multi_a)  # (num_samples, proj_size_mt_audio, t)
            x_multi_v = self.proj_v(x_multi_v)  # (num_samples, proj_size_mt_image, t)
            x_multi_t = x_multi_t.permute(2, 0, 1)  # (t, num_samples, proj_size_mt_text)
            x_multi_a = x_multi_a.permute(2, 0, 1)  # (t, num_samples, proj_size_mt_audio)
            x_multi_v = x_multi_v.permute(2, 0, 1)  # (t, num_samples, proj_size_mt_image)
            x_multi = torch.cat([x_multi_t, x_multi_a, x_multi_v], dim=2)  # (t, num_samples, proj_size_total)
            # x_multi_t = x_multi_t.permute(0, 2, 1)  # (num_samples, t, proj_size)
            # x_multi_a = x_multi_a.permute(0, 2, 1)  # (num_samples, t, proj_size)
            # x_multi_v = x_multi_v.permute(0, 2, 1)  # (num_samples, t, proj_size)
            # x_multi = torch.cat([x_multi_t, x_multi_a, x_multi_v], dim=2)  # (num_samples, t, 3*proj_size)
            # max_list_lens = [max(i, j, k) for i, j, k in zip(text_features_lens, audio_features_lens, img_features_lens)]
            # x_cls_multi, _ = self.transformer_multi(x_multi, max_list_lens, max_all_lens, get_cls=True)
            _, (x_hn_multi, _) = self.rnn_multi(x_multi)
            x_hn_multi = x_hn_multi.transpose(0, 1).flatten(start_dim=1)
            output_multi = self.affine_multi(x_hn_multi)
            output = {"coarse_task": self.final_activation(output_main, self.label_type['coarse_task']),
                      "fine_task": self.final_activation(output_multi, self.label_type['fine_task'])}
        else:
            output = self.final_activation(output_main, self.label_type)

        return output


    @staticmethod
    def get_len_samples(x_feat, mod_name):
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
            #     print("Note from mme2e.py: {}-th feature from modality {} is a zero vector.".format(i, mod_name))
            return real_len

        num_samples, seq_len = x_feat.size(0), x_feat.size(1)
        x = x_feat.numpy()  # (num_samples, seq, feat)
        x_squeezed = squeeze_feat_dim(x)  # (num_samples, seq)
        lens = [get_len_one_sample(x_squeezed[i, :], i) for i in range(num_samples)]
        return lens
