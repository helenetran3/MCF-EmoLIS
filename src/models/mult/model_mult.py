########################################################################################################################
# This file and the 'modules' folder have been copied from the repository of the original authors:
# https://github.com/yaohungt/Multimodal-Transformer
#
# The file has been slightly modified to fit the code of the current repository.
#
# Reference : Tsai, Y. H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L. P., & Salakhutdinov, R. (2019,
# July). Multimodal transformer for unaligned multimodal language sequences. In Proceedings of the conference.
# Association for Computational Linguistics. Meeting (Vol. 2019, p. 6558). NIH Public Access.
########################################################################################################################

import torch
from torch import nn

from src.label_processing import create_label_multitask
from src.models.mult.modules.transformer import TransformerEncoder


class MModel(nn.Module):
    def __init__(self, model_id, param_model, param_task, num_features_text, num_features_image, num_features_audio,
                 num_classes):
        """
        Construct a mult model.
        """
        super(MModel, self).__init__()
        self.model_name = "mult"
        self.model_id = model_id
        self.orig_d_l, self.orig_d_a, self.orig_d_v = num_features_text, num_features_audio, num_features_image
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.d_lav = 30
        self.d_last_hs = 30
        self.num_classes = num_classes
        self.num_scores = 4
        self.vonly = param_model['vonly']
        self.aonly = param_model['aonly']
        self.lonly = param_model['lonly']
        self.num_heads = param_model['num_heads']
        self.layers = param_model['nlevels']
        self.attn_dropout = param_model['attn_dropout']
        self.attn_dropout_a = param_model['attn_dropout_a']
        self.attn_dropout_v = param_model['attn_dropout_v']
        self.attn_dropout_lav = param_model['attn_dropout_lav']
        self.relu_dropout = param_model['relu_dropout']
        self.embed_dropout = param_model['embed_dropout']
        self.res_dropout = param_model['res_dropout']
        self.out_dropout = param_model['out_dropout']
        self.attn_mask = param_model['attn_mask']
        self.dominant_mod = param_model['dominant_modality']
        self.cross_module = param_model['with_cross_module']
        self.lav_as_modality = param_model['lav_as_modality']

        label_type = param_task['label_type']
        self.label_type, _ = create_label_multitask(label_type) if type(label_type) == list else (label_type, None)
        self.multitask = type(self.label_type) == dict

        self.partial_mode = self.lonly + self.aonly + self.vonly

        if self.partial_mode == 0:
            self.lonly = self.vonly = self.aonly = True
            self.partial_mode = 3  # lonly + vonly + aonly

        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v) if not self.dominant_mod else \
                self.d_l + 2 * (self.d_a + self.d_v)

        compute_dim = lambda l: self.num_classes * self.num_scores if 'oh' in l else self.num_classes
        output_dim = [compute_dim(self.label_type['coarse_task']), compute_dim(self.label_type['fine_task'])] \
            if self.multitask else [compute_dim(self.label_type)]

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        if self.lav_as_modality:
            self.proj_lav = nn.Conv1d(self.orig_d_l + self.orig_d_a + self.orig_d_v, self.d_lav,
                                      kernel_size=1, padding=0, bias=False)
        if self.multitask:
            self.proj_last_hs = nn.Conv1d(combined_dim, self.d_last_hs, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly and not self.dominant_mod:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        if self.lav_as_modality:
            self.trans_l_with_lav = self.get_network(self_type='llav')
            self.trans_a_with_lav = self.get_network(self_type='alav')
            self.trans_v_with_lav = self.get_network(self_type='vlav')
            self.trans_lav_with_l = self.get_network(self_type='lavl')
            self.trans_lav_with_a = self.get_network(self_type='lava')
            self.trans_lav_with_v = self.get_network(self_type='lavv')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        if self.lav_as_modality:
            self.trans_lav_mem = self.get_network(self_type='lav_mem', layers=3)
        if self.cross_module:
            self.trans_all_mem = self.get_network(self_type='all_mem', layers=3)
        if self.multitask:
            mod_id = ['l', 'a', 'v']
            self.trans_mem_multi = nn.ModuleList([self.get_network(self_type=mod_id[i]+'_mem_multi', layers=3)
                                                  for i in range(3)])
       
        # Projection layers
        combined_dim = combined_dim*4 if int(self.cross_module + self.lav_as_modality) == 2 else \
            combined_dim*2 if self.lav_as_modality or self.cross_module else combined_dim
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim[0])

        # Multitask
        if self.multitask:
            num_layers = param_model['mt_num_layers']
            dropout = param_model['mt_dropout']
            proj_size_mt_per_mod = param_model['mt_proj_size_per_mod']
            proj_size_total = 3 * proj_size_mt_per_mod
            self.proj_l_mt = nn.Conv1d((self.d_a + self.d_v) + self.d_l, proj_size_mt_per_mod, kernel_size=1, padding=0, bias=False)
            self.proj_a_mt = nn.Conv1d((self.d_l + self.d_v) + self.d_a, proj_size_mt_per_mod, kernel_size=1, padding=0, bias=False)
            self.proj_v_mt = nn.Conv1d((self.d_l + self.d_a) + self.d_v, proj_size_mt_per_mod, kernel_size=1, padding=0, bias=False)

            self.rnn = nn.LSTM(input_size=proj_size_total,
                               hidden_size=proj_size_total,
                               num_layers=num_layers,
                               dropout=(dropout if num_layers > 1 else 0),
                               bidirectional=True)

            self.affine = nn.Sequential(nn.Linear(2*proj_size_total, proj_size_total),
                                        nn.ReLU(),
                                        nn.Linear(proj_size_total, output_dim[1]))


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'lavl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'lava']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'lavv']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type in ['llav', 'alav', 'vlav']:
            embed_dim, attn_dropout = self.d_lav, self.attn_dropout_lav
        elif self_type == 'lav_mem':
            embed_dim, attn_dropout = 3*self.d_lav, self.attn_dropout
        elif self_type == 'l_mem':
            embed_dim = self.d_l if self.dominant_mod else 3*self.d_l if self.lav_as_modality else 2*self.d_l
            attn_dropout = self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim = 3*self.d_a if self.lav_as_modality else 2*self.d_a
            attn_dropout = self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim = 3*self.d_v if self.lav_as_modality else 2*self.d_v
            attn_dropout = self.attn_dropout
        elif self_type == 'all_mem':
            embed_dim = (self.d_l + 2*(self.d_v + self.d_a)) if self.dominant_mod else \
                3*(self.d_v + self.d_a + self.d_l + self.d_lav) if self.lav_as_modality else \
                2*(self.d_v + self.d_a + self.d_l)
            attn_dropout = self.attn_dropout
        elif self_type == 'l_mem_multi':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem_multi':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem_multi':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def mult_backbone(self, x_l, x_a, x_v):

        x_l = torch.dropout(x_l.transpose(1, 2), p=self.embed_dropout, train=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        h_all_s = []  # Will contain all outputs of crossmodal transformers, to concatenate and create cross-module

        if self.lav_as_modality:
            x_lav = torch.cat([x_l, x_a, x_v], dim=1)
            proj_x_lav = self.proj_lav(x_lav)
            proj_x_lav = proj_x_lav.permute(2, 0, 1)
            # proj_x_concat = torch.cat([proj_x_l, proj_x_a, proj_x_v, proj_x_lav], dim=2)
            h_lav_with_ls = self.trans_lav_with_l(proj_x_lav, proj_x_l, proj_x_l)    # Dimension (L, N, d_lav)
            h_lav_with_as = self.trans_lav_with_a(proj_x_lav, proj_x_a, proj_x_a)    # Dimension (L, N, d_lav)
            h_lav_with_vs = self.trans_lav_with_v(proj_x_lav, proj_x_v, proj_x_v)    # Dimension (L, N, d_lav)
            h_lavs = [h_lav_with_ls, h_lav_with_as, h_lav_with_vs]
            if self.cross_module:
                h_all_s += h_lavs
            h_lavs = torch.cat(h_lavs, dim=2)
            h_lavs = self.trans_lav_mem(h_lavs)

            if type(h_lavs) == tuple:
                h_lavs = h_lavs[0]
            last_h_lav = last_hs = h_lavs[-1]   # Take the last output for prediction

        if self.lonly:
            # (V,A) --> L
            if self.dominant_mod:
                if self.cross_module:
                    h_all_s += [proj_x_l]
                h_ls = self.trans_l_mem(proj_x_l)
            else:
                h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
                h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
                if self.cross_module:
                    h_all_s += [h_l_with_as, h_l_with_vs]
                if self.lav_as_modality:
                    h_l_with_lavs = self.trans_l_with_lav(proj_x_l, proj_x_lav, proj_x_lav)
                    h_all_s += [h_l_with_lavs]
                h_ls = [h_l_with_as, h_l_with_vs, h_l_with_lavs] if self.lav_as_modality else [h_l_with_as, h_l_with_vs]
                h_ls = torch.cat(h_ls, dim=2)
                h_ls = self.trans_l_mem(h_ls)

            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            if self.cross_module:
                h_all_s += [h_a_with_ls, h_a_with_vs]
            if self.lav_as_modality:
                h_a_with_lavs = self.trans_a_with_lav(proj_x_a, proj_x_lav, proj_x_lav)
                h_all_s += [h_a_with_lavs]
            h_as = [h_a_with_ls, h_a_with_vs, h_a_with_lavs] if self.lav_as_modality else [h_a_with_ls, h_a_with_vs]
            h_as = torch.cat(h_as, dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            if self.cross_module:
                h_all_s += [h_v_with_ls, h_v_with_as]
            if self.lav_as_modality:
                h_v_with_lavs = self.trans_v_with_lav(proj_x_v, proj_x_lav, proj_x_lav)
                h_all_s += [h_v_with_lavs]
            h_vs = [h_v_with_ls, h_v_with_as, h_v_with_lavs] if self.lav_as_modality else [h_v_with_ls, h_v_with_as]
            h_vs = torch.cat(h_vs, dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        if self.cross_module:
            h_all_s = torch.cat(h_all_s, dim=2)
            h_all_s = self.trans_all_mem(h_all_s)
            if type(h_all_s) == tuple:
                h_all_s = h_all_s[0]
            last_h_all = last_hs = h_all_s[-1]

        if self.partial_mode == 3:
            last_h_list = [last_h_l, last_h_a, last_h_v]
            if self.lav_as_modality:
                last_h_list += [last_h_lav]
            if self.cross_module:
                last_h_list += [last_h_all]
            last_hs = last_h_list
            #last_hs = torch.cat(last_h_list, dim=1)

        proj_x = [proj_x_l, proj_x_a, proj_x_v]
        hs_full_seq = [h_ls, h_as, h_vs]

        return proj_x, last_hs, hs_full_seq

    def final_activation(self, output, label_i):
        if label_i in ['present', 'dominant', 'coarse_prob']:
            return torch.sigmoid(output)
        elif 'oh' in label_i:
            return torch.softmax(output.view(-1, self.num_classes, self.num_scores), dim=2)
        else:
            return torch.relu(output)

    def forward(self, x_l, x_v, x_a):  # inverted x_v and x_a to fit the current code
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """

        proj_x, last_hs, hs_full_seq = self.mult_backbone(x_l, x_a, x_v)
        last_hs = torch.cat(last_hs, dim=1)

        # Residual block for main task
        last_hs_proj = self.proj2(torch.dropout(torch.relu(self.proj1(last_hs)),
                                                p=self.out_dropout, train=self.training))
        last_hs_proj += last_hs
        output_main = self.out_layer(last_hs_proj)

        if self.multitask:
            # # Apply self-attention transformers to all projected vectors
            # proj_mem = [self.trans_mem_multi[i](proj_x[i]) for i in range(3)]

            # Reduce the full hidden sequence of each modality to proj_size along the num_features axis
            x_l_multi = torch.cat((proj_x[0], hs_full_seq[0]), 2)
            x_a_multi = torch.cat((proj_x[1], hs_full_seq[1]), 2)
            x_v_multi = torch.cat((proj_x[2], hs_full_seq[2]), 2)
            x_l_multi = x_l_multi.permute(1, 2, 0)  # (num_samples, num_features, t)
            x_a_multi = x_a_multi.permute(1, 2, 0)  # (num_samples, num_features, t)
            x_v_multi = x_v_multi.permute(1, 2, 0)  # (num_samples, num_features, t)
            x_l_multi = self.proj_l_mt(x_l_multi)  # (num_samples, proj_size, t)
            x_a_multi = self.proj_a_mt(x_a_multi)  # (num_samples, proj_size, t)
            x_v_multi = self.proj_v_mt(x_v_multi)  # (num_samples, proj_size, t)
            x_l_multi = x_l_multi.permute(2, 0, 1)  # (t, num_samples, proj_size)
            x_a_multi = x_a_multi.permute(2, 0, 1)  # (t, num_samples, proj_size)
            x_v_multi = x_v_multi.permute(2, 0, 1)  # (t, num_samples, proj_size)
            x_multi = torch.cat((x_l_multi, x_a_multi, x_v_multi), 2)

            # Apply RNN + FC layers to the concatenated vectors
            _, (x_hn_multi, _) = self.rnn(x_multi)
            x_hn_multi = x_hn_multi.transpose(0, 1).flatten(start_dim=1)
            output_multi = self.affine(x_hn_multi)
            output = {"coarse_task": self.final_activation(output_main, self.label_type['coarse_task']),
                      "fine_task": self.final_activation(output_multi, self.label_type['fine_task'])}
        else:
            output = self.final_activation(output_main, self.label_type)

        return output#, last_hs_1
