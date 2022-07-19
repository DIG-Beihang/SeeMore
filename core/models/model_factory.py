import os
import torch
from torch.optim import Adam
from core.models import seemore
from core.models.meta import Meta


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'seemore': seemore.SeeMore,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        
        if configs.training_stage == 1:
            self.optimizer = Adam(self.network.parameters(), lr=configs.lr_alpha)
        elif configs.training_stage == 2:
            self.maml = Meta(self.network, configs).to(configs.device)
            self.update_lr_flag = True

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'], strict=True)

    def train(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        if self.configs.training_stage == 1:
            self.optimizer.zero_grad()
            next_frames, (loss, loss_old, loss_rev, loss_bilstm) = self.network(frames_tensor, mask_tensor)
            loss.backward()
            self.optimizer.step()
            return loss.detach().cpu().numpy()
        elif self.configs.training_stage == 2:
            meta_loss = self.maml(frames_tensor, mask_tensor, update_lr=self.update_lr_flag)
            return meta_loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _, _ = self.network.rnn(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()
