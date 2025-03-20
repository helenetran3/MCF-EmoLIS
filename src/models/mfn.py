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

from src.label_processing import create_label_multitask


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
        label_type = param_task['label_type']

        self.label_type, _ = create_label_multitask(label_type) if type(label_type) == list else (label_type, None)
        self.multitask = type(self.label_type) == dict
        self.num_classes = num_classes  # We changed 1 to num_classes
        self.num_scores = 4

        compute_dim = lambda l: self.num_classes * self.num_scores if 'oh' in l else self.num_classes
        output_dim = [compute_dim(self.label_type['coarse_task']), compute_dim(self.label_type['fine_task'])] \
            if self.multitask else [compute_dim(self.label_type)]

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

        num_branches = 2 if self.multitask else 1

        d_l_input = [self.d_l, self.d_l + self.dh_l + self.mem_dim] if self.multitask else [self.d_l]
        d_a_input = [self.d_a, self.d_a + self.dh_a + self.mem_dim] if self.multitask else [self.d_a]
        d_v_input = [self.d_v, self.d_v + self.dh_v + self.mem_dim] if self.multitask else [self.d_v]
        self.lstm_l = nn.ModuleList([nn.LSTMCell(d_l_input[i], self.dh_l) for i in range(num_branches)])
        self.lstm_a = nn.ModuleList([nn.LSTMCell(d_a_input[i], self.dh_a) for i in range(num_branches)])
        self.lstm_v = nn.ModuleList([nn.LSTMCell(d_v_input[i], self.dh_v) for i in range(num_branches)])

        self.att1_fc1 = nn.ModuleList([nn.Linear(attInShape, h_att1) for _ in range(num_branches)])
        self.att1_fc2 = nn.ModuleList([nn.Linear(h_att1, attInShape) for _ in range(num_branches)])
        self.att1_dropout = nn.ModuleList([nn.Dropout(att1_dropout) for _ in range(num_branches)])

        self.att2_fc1 = nn.ModuleList([nn.Linear(attInShape, h_att2) for _ in range(num_branches)])
        self.att2_fc2 = nn.ModuleList([nn.Linear(h_att2, self.mem_dim) for _ in range(num_branches)])
        self.att2_dropout = nn.ModuleList([nn.Dropout(att2_dropout) for _ in range(num_branches)])

        self.gamma1_fc1 = nn.ModuleList([nn.Linear(gammaInShape, h_gamma1) for _ in range(num_branches)])
        self.gamma1_fc2 = nn.ModuleList([nn.Linear(h_gamma1, self.mem_dim) for _ in range(num_branches)])
        self.gamma1_dropout = nn.ModuleList([nn.Dropout(gamma1_dropout) for _ in range(num_branches)])

        self.gamma2_fc1 = nn.ModuleList([nn.Linear(gammaInShape, h_gamma2) for _ in range(num_branches)])
        self.gamma2_fc2 = nn.ModuleList([nn.Linear(h_gamma2, self.mem_dim) for _ in range(num_branches)])
        self.gamma2_dropout = nn.ModuleList([nn.Dropout(gamma2_dropout) for _ in range(num_branches)])

        self.out_fc1 = nn.ModuleList([nn.Linear(final_out, h_out) for _ in range(num_branches)])
        self.out_fc2 = nn.ModuleList([nn.Linear(h_out, output_dim[i]) for i in range(num_branches)])
        self.out_dropout = nn.ModuleList([nn.Dropout(out_dropout) for _ in range(num_branches)])

    def mfn_backbone(self, x_l, x_v, x_a, n, t, ind):
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
            new_h_l, new_c_l = self.lstm_l[ind](x_l[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a[ind](x_a[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v[ind](x_v[i], (self.h_v, self.c_v))
            # concatenate
            prev_cs = torch.cat([prev_c_l, prev_c_a, prev_c_v], dim=1)
            new_cs = torch.cat([new_c_l, new_c_a, new_c_v], dim=1)
            cStar = torch.cat([prev_cs, new_cs], dim=1)
            attention = torch.softmax(self.att1_fc2[ind](self.att1_dropout[ind](torch.relu(self.att1_fc1[ind](cStar)))), dim=1)
            attended = attention * cStar
            cHat = torch.tanh(self.att2_fc2[ind](self.att2_dropout[ind](torch.relu(self.att2_fc1[ind](attended)))))
            both = torch.cat([attended, self.mem], dim=1)
            gamma1 = torch.sigmoid(self.gamma1_fc2[ind](self.gamma1_dropout[ind](torch.relu(self.gamma1_fc1[ind](both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2[ind](self.gamma2_dropout[ind](torch.relu(self.gamma2_fc1[ind](both)))))
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

        return all_mems, all_h_ls, all_h_as, all_h_vs, all_c_ls, all_c_as, all_c_vs

    def final_activation(self, output, label_i):
        if label_i in ['present', 'dominant']:
            return torch.sigmoid(output)
        elif 'oh' in label_i:
            return torch.softmax(output.view(-1, self.num_classes, self.num_scores), dim=2)
        else:
            return torch.relu(output)

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

        all_mems, all_h_ls, all_h_as, all_h_vs, all_c_ls, all_c_as, all_c_vs = \
            self.mfn_backbone(x_l, x_v, x_a, n, t, ind=0)
        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_mem], dim=1)
        output_main = self.out_fc2[0](self.out_dropout[0](torch.relu(self.out_fc1[0](last_hs))))

        # last hidden layer last_hs is n x h
        if self.multitask:
            # Concatenate input sequences + hidden cells and cross-module from first branch
            x_l_multi = torch.cat((x_l, torch.stack(all_h_ls), torch.stack(all_mems)), 2)
            x_a_multi = torch.cat((x_a, torch.stack(all_h_as), torch.stack(all_mems)), 2)
            x_v_multi = torch.cat((x_v, torch.stack(all_h_vs), torch.stack(all_mems)), 2)
            all_mems_multi, all_h_ls_multi, all_h_as_multi, all_h_vs_multi, _, _, _ = \
                self.mfn_backbone(x_l_multi, x_v_multi, x_a_multi, n, t, ind=1)
            last_h_l_multi = all_h_ls_multi[-1]
            last_h_a_multi = all_h_as_multi[-1]
            last_h_v_multi = all_h_vs_multi[-1]
            last_mem_multi = all_mems_multi[-1]
            last_hs_multi = torch.cat([last_h_l_multi, last_h_a_multi, last_h_v_multi, last_mem_multi], dim=1)
            output_multi = self.out_fc2[1](self.out_dropout[1](torch.relu(self.out_fc1[1](last_hs_multi))))
            output = {"coarse_task": self.final_activation(output_main, self.label_type['coarse_task']),
                      "fine_task": self.final_activation(output_multi, self.label_type['fine_task'])}
        else:
            output = self.final_activation(output_main, self.label_type)

        return output
