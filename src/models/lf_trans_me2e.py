import numpy as np
import torch
from torch import nn
from typing import List
from src.models.me2e.modules.transformer_encoder import WrappedTransformerEncoder


class MModel(nn.Module):
    def __init__(self, model_id, param_model, num_features_text, num_features_image, num_features_audio, num_classes):
        super(MModel, self).__init__()

        self.model_name = 'lf_rnn_me2e'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio
        self.mods = 'tav'  # we use the three modalities here

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

        self.affines = nn.ModuleDict({
            't': nn.Sequential(
                nn.Linear(num_features_text, num_features_text // 2),
                nn.ReLU(),
                nn.Linear(num_features_text // 2, num_classes),
                nn.Sigmoid()
            ),
            'a': nn.Sequential(
                nn.Linear(num_features_audio, num_features_audio // 2),
                nn.ReLU(),
                nn.Linear(num_features_audio // 2, num_classes),
                nn.Sigmoid()
            ),
            'v': nn.Sequential(
                nn.Linear(num_features_image, num_features_image // 2),
                nn.ReLU(),
                nn.Linear(num_features_image // 2, num_classes),
                nn.Sigmoid()
            )
        })

        self.weighted_fusion = nn.Sequential(nn.Linear(len(self.mods), 1, bias=False),
                                             nn.Sigmoid())

    def forward(self, x_text, x_image, x_audio):
        all_logits = []
        audio_features_lens = self.get_len_samples(x_audio, 'audio')
        img_features_lens = self.get_len_samples(x_image, 'image')

        if 't' in self.mods:
            texts = self.transformers['t'](x_text, get_cls=True)
            texts = self.affines['t'](texts)
            all_logits.append(texts)

        if 'a' in self.mods:
            audio_features = self.transformers['a'](x_audio, audio_features_lens, get_cls=True)
            audio_features = self.affines['a'](audio_features)
            all_logits.append(audio_features)

        if 'v' in self.mods:
            img_features = self.transformers['v'](x_image, img_features_lens, get_cls=True)
            img_features = self.affines['v'](img_features)
            all_logits.append(img_features)

        if len(self.mods) == 1:
            return all_logits[0]

        return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)

    @staticmethod
    def get_len_samples(x_feat, mod_name):
        """
        Return a list of the sequence length of each sample from x_feat, based on the last index with non-zero value.

        x_feat: array of shape (num_samples, num_features) where padding has already been performed.
        mod_name: name of the modality ('image' or 'audio')
        """

        def get_len_one_sample(x_feat_i, i):
            real_len = np.max(np.nonzero(x_feat_i)) + 1  # +1 because we want the length
            if real_len == 0:
                print("Note from mme2e.py: {}-th feature from modality {} is a zero vector.".format(i, mod_name))
            return real_len

        num_samples = x_feat.size(0)
        x = x_feat.numpy()
        lens = [get_len_one_sample(x[i, :], i) for i in range(num_samples)]
        return lens
