import numpy as np
import torch
from torch import nn
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
        num_features_tav_incr = num_features_text + num_features_image + num_features_audio + 1 # 409+1 for div by 5

        self.concatenate = torch.cat

        self.transformer = WrappedTransformerEncoder(dim=num_features_tav_incr,
                                                     num_layers=nlayers,
                                                     num_heads=5)

        self.affine = nn.Sequential(nn.Linear(num_features_tav_incr, num_features_tav_incr // 2),
                                    nn.ReLU(),
                                    nn.Linear(num_features_tav_incr // 2, num_classes),
                                    nn.Sigmoid())

    def forward(self, x_text, x_image, x_audio):

        max_all_lens = x_text.size(1)
        assert max_all_lens == x_image.size(1) == x_audio.size(1)
        num_samples = x_text.size(0)
        assert num_samples == x_image.size(0) == x_audio.size(0)

        zero_vector = torch.zeros(num_samples, max_all_lens, 1)
        x = self.concatenate((x_text, x_image, x_audio, zero_vector), 2)
        feat_lens = self.get_len_samples(x)
        x_trans = self.transformers(x, feat_lens, max_all_lens, get_cls=True)
        x_aff = self.affines(x_trans)

        return x_aff

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
