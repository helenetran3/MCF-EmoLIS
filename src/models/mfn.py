########################################################################################################################
# The code of this file have been copied from the repository of the original authors:
# https://github.com/pliang279/MFN
#
# The file has been slightly modified to fit the code of the current repository.
#
# Reference : Zadeh, A., Liang, P. P., Mazumder, N., Poria, S., Cambria, E., & Morency, L. P. (2018, April).
# Memory Fusion Network for Multi-view Sequential Learning. In Proceedings of the AAAI conference on artificial
# intelligence (Vol. 32, No. 1).
########################################################################################################################

import torch
from torch import nn


class MModel(nn.Module):
    # def __init__(self, config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig):
    def __init__(self, model_id, param_model, param_task, num_features_text,  num_features_image, num_features_audio,
                 num_classes):
        super(MModel, self).__init__()
        [self.d_l, self.d_v, self.d_a] = [num_features_text,  num_features_image, num_features_audio]
        [self.dh_l, self.dh_v, self.dh_a] = [param_model['dim_h_text'], param_model['dim_h_image'],
                                             param_model['dim_h_audio']]
        total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.mem_dim = param_model["memsize"]
        window_dim = param_model["windowsize"]
        self.gml = param_model['gml']
        self.label_type = param_task['label_type']
        self.num_classes = num_classes  # We changed 1 to num_classes
        self.num_scores = 4
        output_dim = self.num_classes * self.num_scores if self.gml else self.num_classes

        attInShape = total_h_dim * window_dim
        gammaInShape = attInShape + self.mem_dim
        final_out = total_h_dim + self.mem_dim
        h_att1 = param_model['nn1_shapes']
        h_att2 = param_model['nn2_shapes']
        h_gamma1 = param_model['gamma1_shapes']
        h_gamma2 = param_model['gamma2_shapes']
        h_out = param_model['out_shapes']
        att1_dropout = param_model['nn1_drop']
        att2_dropout = param_model['nn2_drop']
        gamma1_dropout = param_model['gamma1_drop']
        gamma2_dropout = param_model['gamma2_drop']
        out_dropout = param_model['out_drop']

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, x_l, x_v, x_a):
        # x_l = x[:, :, :self.d_l]
        # x_a = x[:, :, self.d_l:self.d_l + self.d_a]
        # x_v = x[:, :, self.d_l + self.d_a:]
        # x is t x n x d
        ### the three following lines is to match with the required dimensions
        x_l = x_l.transpose(0, 1)
        x_a = x_a.transpose(0, 1)
        x_v = x_v.transpose(0, 1)
        n = x_l.shape[1]
        t = x_l.shape[0]
        self.h_l = torch.zeros(n, self.dh_l)#.cuda()
        self.h_a = torch.zeros(n, self.dh_a)#.cuda()
        self.h_v = torch.zeros(n, self.dh_v)#.cuda()
        self.c_l = torch.zeros(n, self.dh_l)#.cuda()
        self.c_a = torch.zeros(n, self.dh_a)#.cuda()
        self.c_v = torch.zeros(n, self.dh_v)#.cuda()
        self.mem = torch.zeros(n, self.mem_dim)#.cuda()
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_mems = []
        for i in range(t):
            # prev time step
            prev_c_l = self.c_l
            prev_c_a = self.c_a
            prev_c_v = self.c_v
            # curr time step
            new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
            # concatenate
            prev_cs = torch.cat([prev_c_l, prev_c_a, prev_c_v], dim=1)
            new_cs = torch.cat([new_c_l, new_c_a, new_c_v], dim=1)
            cStar = torch.cat([prev_cs, new_cs], dim=1)
            attention = torch.softmax(self.att1_fc2(self.att1_dropout(torch.relu(self.att1_fc1(cStar)))), dim=1)
            attended = attention * cStar
            cHat = torch.tanh(self.att2_fc2(self.att2_dropout(torch.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended, self.mem], dim=1)
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(torch.relu(self.gamma1_fc1(both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(torch.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1 * self.mem + gamma2 * cHat
            all_mems.append(self.mem)
            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)

        # last hidden layer last_hs is n x h
        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_mem], dim=1)
        output = self.out_fc2(self.out_dropout(torch.relu(self.out_fc1(last_hs))))

        if self.gml:
            output = output.view(-1, self.num_classes, self.num_scores)
            output = torch.softmax(output, dim=2)
        else:
            if self.label_type in ['present', 'dominant']:
                output = torch.sigmoid(output)
            else:
                output = torch.relu(output)

        return output
