import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_v2 import SpatioTemporalLSTMCell
import torch.nn.functional as F
from collections import OrderedDict


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs

        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, params=None, prefix='rnn.', test_flag=False):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        next_frames_feat = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        total_length = self.configs.total_length
        if test_flag:
            total_length = self.configs.total_length_test

        for t in range(total_length - 1):
            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            if params == None:
                h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
                delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

                for i in range(1, self.num_layers):
                    h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                    delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                    delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

                x_gen = self.conv_last(h_t[self.num_layers - 1])
            else:
                h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory, params, 0, prefix)
                adapt_delta_c = F.conv2d(
                    delta_c,
                    params[prefix+'adapter.weight'],
                    stride=1,
                    padding=0)
                adapt_delta_m = F.conv2d(
                    delta_m,
                    params[prefix+'adapter.weight'],
                    stride=1,
                    padding=0)
                delta_c_list[0] = F.normalize(adapt_delta_c.view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[0] = F.normalize(adapt_delta_m.view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                
                for i in range(1, self.num_layers):
                    h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory, params, i, prefix)
                    adapt_delta_c = F.conv2d(
                        delta_c,
                        params[prefix+'adapter.weight'],
                        stride=1,
                        padding=0)
                    adapt_delta_m = F.conv2d(
                        delta_m,
                        params[prefix+'adapter.weight'],
                        stride=1,
                        padding=0)
                    delta_c_list[i] = F.normalize(adapt_delta_c.view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                    delta_m_list[i] = F.normalize(adapt_delta_m.view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                
                x_gen = F.conv2d(
                        h_t[self.num_layers - 1],
                        params[prefix+'conv_last.weight'],
                        stride=1,
                        padding=0)
            
            next_frames.append(x_gen)
            next_frames_feat.append(h_t[self.num_layers - 1])

            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames_feat = torch.stack(next_frames_feat, dim=1).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + self.configs.decouple_beta * decouple_loss
        
        return next_frames, next_frames_feat, loss


class SeeMore(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(SeeMore, self).__init__()
        self.configs = configs
        self.rnn = RNN(num_layers, num_hidden, configs)
        self.rnn_rev = RNN(num_layers, num_hidden, configs)
        self.MSE_criterion = nn.MSELoss()

        if configs.training_stage == 2:
            self.task_lr = OrderedDict()
            for key in range(num_layers):
                self.task_lr[str(key)] = nn.Parameter(configs.lr_alpha * torch.ones([1], requires_grad=True).to(configs.device))
            for key in ['conv_last', 'adapter']:
                self.task_lr[str(key)] = nn.Parameter(configs.lr_alpha * torch.ones([1], requires_grad=True).to(configs.device))
            
            self.task_lr_rev = OrderedDict()
            for key in range(num_layers):
                self.task_lr_rev[str(key)] = nn.Parameter(configs.lr_alpha * torch.ones([1], requires_grad=True).to(configs.device))
            for key in ['conv_last', 'adapter']:
                self.task_lr_rev[str(key)] = nn.Parameter(configs.lr_alpha * torch.ones([1], requires_grad=True).to(configs.device))
    
    def cloned_state_dict(self):
        """
        Only returns state_dict of meta_learner (not task_lr)
        """
        cloned_state_dict = {
            key: val.clone() for key, val in self.state_dict().items()
        }
        return cloned_state_dict

    def forward(self, frames_tensor, mask_true, params=None):
        next_frames, next_frames_feat, loss_old = self.rnn(frames_tensor, mask_true, params, prefix='rnn.')
        next_frames_rev, next_frames_feat_rev, loss_rev = self.rnn_rev(frames_tensor.flip(dims=[1]), mask_true, params, prefix='rnn_rev.')
        
        next_frames_rev = next_frames_rev.flip(dims=[1])
        next_frames_feat_rev = next_frames_feat_rev.flip(dims=[1])
        loss_bilstm = self.MSE_criterion(next_frames_feat[:, :-1], next_frames_feat_rev[:, 1:]) + self.MSE_criterion(next_frames[:, :-1], next_frames_rev[:, 1:])
        
        loss = loss_old + loss_rev + loss_bilstm
        
        return next_frames, (loss, loss_old, loss_rev, loss_bilstm)
