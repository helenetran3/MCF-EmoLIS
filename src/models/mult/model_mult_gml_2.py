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

from src.models.mult.modules.transformer import TransformerEncoder


class MModel(nn.Module):
    def __init__(self, model_id, param_model, param_task, num_features_text, num_features_image, num_features_audio,
                 num_classes):
        """
        Construct a mult model.
        """
        super(MModel, self).__init__()
        self.model_name = "mult_gml_2"
        self.model_id = model_id
        self.orig_d_l, self.orig_d_a, self.orig_d_v = num_features_text, num_features_audio, num_features_image
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.vonly = param_model['vonly']
        self.aonly = param_model['aonly']
        self.lonly = param_model['lonly']
        self.num_heads = param_model['num_heads']
        self.layers = param_model['nlevels']
        self.attn_dropout = param_model['attn_dropout']
        self.attn_dropout_a = param_model['attn_dropout_a']
        self.attn_dropout_v = param_model['attn_dropout_v']
        self.relu_dropout = param_model['relu_dropout']
        self.embed_dropout = param_model['embed_dropout']
        self.res_dropout = param_model['res_dropout']
        self.out_dropout = param_model['out_dropout']
        self.attn_mask = param_model['attn_mask']
        self.num_classes = num_classes
        self.num_scores = 4

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly

        if self.partial_mode == 0:
            self.lonly = self.vonly = self.aonly = True
            self.partial_mode = 3  # lonly + vonly + aonly

        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        self.output_dim = self.num_scores

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem, self.trans_a_mem, self.trans_v_mem = [], [], []
        for _ in range(self.num_classes):
            self.trans_l_mem += [self.get_network(self_type='l_mem', layers=3)]
            self.trans_a_mem += [self.get_network(self_type='a_mem', layers=3)]
            self.trans_v_mem += [self.get_network(self_type='v_mem', layers=3)]
       
        # Projection layers
        self.proj_1, self.proj_2, self.out_layer = [], [], []
        for _ in range(self.num_classes):
            self.proj_1 += [nn.Linear(combined_dim, combined_dim)]
            self.proj_2 += [nn.Linear(combined_dim, combined_dim)]
            self.out_layer += [nn.Linear(combined_dim, self.output_dim)]

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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
            
    def forward(self, x_l, x_v, x_a):  # inverted x_v and x_a to fit the current code
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
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

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls_classes = []
            for i in range(self.num_classes):
                h_ls_i = self.trans_l_mem[i](h_ls)
                h_ls_i = h_ls_i[0] if type(h_ls_i) == tuple else h_ls_i
                h_ls_classes += [h_ls_i]
            last_h_l = last_hs = [h_ls_i[-1] for h_ls_i in h_ls_classes]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as_classes = []
            for i in range(self.num_classes):
                h_as_i = self.trans_a_mem[i](h_as)
                h_as_i = h_as_i[0] if type(h_as_i) == tuple else h_as_i
                h_as_classes += [h_as_i]
            last_h_a = last_hs = [h_as_i[-1] for h_as_i in h_as_classes]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs_classes = []
            for i in range(self.num_classes):
                h_vs_i = self.trans_v_mem[i](h_vs)
                h_vs_i = h_vs_i[0] if type(h_vs_i) == tuple else h_vs_i
                h_vs_classes += [h_vs_i]
            last_h_v = last_hs = [h_vs_i[-1] for h_vs_i in h_vs_classes]
        
        if self.partial_mode == 3:
            last_hs = []
            for last_h_l_i, last_h_a_i, last_h_v_i in zip(last_h_l, last_h_a, last_h_v):
                last_hs += [torch.cat([last_h_l_i, last_h_a_i, last_h_v_i], dim=1)]
        
        # A residual block
        output = []
        for i in range(self.num_classes):
            last_hs_proj_i = self.proj_2[i](torch.dropout(torch.relu(self.proj_1[i](last_hs[i])), p=self.out_dropout,
                                                      train=self.training))
            last_hs_proj_i += last_hs[i]
            output += [torch.softmax(self.out_layer[i](last_hs_proj_i), dim = 1)]

        output = [o.unsqueeze(-1) for o in output]
        output = torch.concat(output, axis=-1)  # (batch_size, num_scores, num_classes)
        output = output.permute(0, 2, 1)  # (batch_size, num_classes, num_scores)
        return output#, last_hs
