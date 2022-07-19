import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import OrderedDict


class Meta(nn.Module):
    def __init__(self, network, configs):
        super(Meta, self).__init__()
        self.configs = configs
        self.network = network
        self.update_step = 1

        model_params = list(self.network.task_lr.values()) + list(self.network.task_lr_rev.values())
        self.meta_opt = Adam(model_params, lr=configs.lr_beta)

        self.inner_vs = OrderedDict()
        self.inner_sqrs = OrderedDict()

        for key, param in self.network.named_parameters():
            self.inner_vs[key] = torch.zeros_like(param.data)
            self.inner_sqrs[key] = torch.zeros_like(param.data)
        
        self.inner_hparams = {'t': 1}

        self.meta_vs = OrderedDict()
        self.meta_sqrs = OrderedDict()
        for key, param in self.network.named_parameters():
            self.meta_vs[key] = torch.zeros_like(param.data)
            self.meta_sqrs[key] = torch.zeros_like(param.data)
        self.meta_hparams = {'t': 1}

        self.eps = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999

    def train_model_lr(self, x_spt, x_spt_mask_tensor):
        self.network.train()

        next_frames, (loss, loss_old, loss_rev, loss_bilstm) = self.network(x_spt, x_spt_mask_tensor)

        def zero_grad(params):
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

        zero_grad(self.network.parameters())
        grads = torch.autograd.grad(loss, self.network.parameters(), create_graph=True)

        adapted_state_dict = self.network.cloned_state_dict()
        adapted_params = OrderedDict()
        for (key, param), grad in zip(self.network.named_parameters(), grads):
            if key.find('_rev') == -1:
                if key.split('.')[1] == 'conv_last' or key.split('.')[1] == 'adapter':
                    layer = key.split('.')[1]
                else:
                    layer = key.split('.')[2]
                task_lr = self.network.task_lr[layer]
            else:
                if key.split('.')[1] == 'conv_last' or key.split('.')[1] == 'adapter':
                    layer = key.split('.')[1]
                else:
                    layer = key.split('.')[2]
                task_lr = self.network.task_lr_rev[layer]

            self.inner_vs[key][:] = self.beta1 * self.inner_vs[key] + (1 - self.beta1) * grad.data
            self.inner_sqrs[key][:] = self.beta2 * self.inner_sqrs[key] + (1 - self.beta2) * grad.data ** 2
            v_hat = self.inner_vs[key] / (1 - self.beta1 ** self.inner_hparams['t'])
            s_hat = self.inner_sqrs[key] / (1 - self.beta2 ** self.inner_hparams['t'])
            adapted_params[key] = param.data - task_lr * v_hat / (torch.sqrt(s_hat) + self.eps)
            adapted_state_dict[key] = adapted_params[key]

        zero_grad(self.network.parameters())

        self.inner_hparams['t'] += 1
        return adapted_state_dict

    def train_task_param(self, frames_tensor, mask_tensor):
        self.network.train()

        next_frames, (loss, loss_old, loss_rev, loss_bilstm) = self.network(frames_tensor, mask_tensor)

        def zero_grad(params):
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

        zero_grad(self.network.parameters())
        loss.backward()

        for key, param in self.network.named_parameters():
            if  key.find('_rev') == -1:
                if key.split('.')[1] == 'conv_last' or key.split('.')[1] == 'adapter':
                    layer = key.split('.')[1]
                else:
                    layer = key.split('.')[2]
                task_lr = self.network.task_lr[layer]
            
            else:
                if key.split('.')[1] == 'conv_last' or key.split('.')[1] == 'adapter':
                    layer = key.split('.')[1]
                else:
                    layer = key.split('.')[2]
                task_lr = self.network.task_lr_rev[layer]

            self.meta_vs[key][:] = self.beta1 * self.meta_vs[key] + (1 - self.beta1) * param.grad.data
            self.meta_sqrs[key][:] = self.beta2 * self.meta_sqrs[key] + (1 - self.beta2) * param.grad.data ** 2
            v_hat = self.meta_vs[key] / (1 - self.beta1 ** self.meta_hparams['t'])
            s_hat = self.meta_sqrs[key] / (1 - self.beta2 ** self.meta_hparams['t'])
            
            param.data = param.data - task_lr.data * v_hat / (torch.sqrt(s_hat) + self.eps)
        
        self.meta_hparams['t'] += 1
        return next_frames, loss

    def forward(self, frames_tensor, mask_tensor, update_lr_flag=False):
        if not update_lr_flag:
            next_frames, loss = self.train_task_param(frames_tensor, mask_tensor)
            return loss

        self.meta_opt.zero_grad()
        updated_para_dict = self.train_model_lr(frames_tensor, mask_tensor)
        next_frames, (meta_loss, loss_old, loss_rev, loss_bilstm) = self.network(frames_tensor, mask_tensor, updated_para_dict)

        lambd = 2
        incre_loss = 10*(F.relu(lambd*self.network.task_lr['0'] - self.network.task_lr['1']) + \
            F.relu(lambd*self.network.task_lr['1'] - self.network.task_lr['2']) + \
                 F.relu(lambd*self.network.task_lr['2'] - self.network.task_lr['3']) + \
                     F.relu(lambd*self.network.task_lr['3'] - self.network.task_lr['conv_last'])) 

        incre_loss_rev = 10*(F.relu(lambd*self.network.task_lr_rev['0'] - self.network.task_lr_rev['1']) + \
            F.relu(lambd*self.network.task_lr_rev['1'] - self.network.task_lr_rev['2']) + \
                 F.relu(lambd*self.network.task_lr_rev['2'] - self.network.task_lr_rev['3']) + \
                     F.relu(lambd*self.network.task_lr_rev['3'] - self.network.task_lr_rev['conv_last'])) 

        meta_loss += incre_loss[0] + incre_loss_rev[0]

        meta_loss.backward()
        self.meta_opt.step()

        for key, param in self.network.named_parameters():
            if key.find('_rev') == -1:
                if key.split('.')[1] == 'conv_last' or key.split('.')[1] == 'adapter':
                    layer = key.split('.')[1]
                else:
                    layer = key.split('.')[2]
                task_lr = self.network.task_lr[layer]
            else:
                if key.split('.')[1] == 'conv_last' or key.split('.')[1] == 'adapter':
                    layer = key.split('.')[1]
                else:
                    layer = key.split('.')[2]
                task_lr = self.network.task_lr_rev[layer]

            task_lr.data[task_lr.data < 0] = 0
            task_lr.data[task_lr.data > 0.0002] = 0.0002

        return meta_loss
