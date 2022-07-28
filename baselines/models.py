import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import argparse


class MLP(nn.Module):
    def __init__(self, agent_config):
        super(MLP, self).__init__()
        self.args = argparse.Namespace(**agent_config)
        self.hid_size = self.args.hid_size
        self.obs_shape = self.args.obs_shape
        self.n_actions = self.args.n_actions

        self.affine1 = nn.Linear(self.obs_shape, self.hid_size)
        self.affine2 = nn.Linear(self.hid_size, self.hid_size)
        self.head = nn.Linear(self.hid_size,self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)
        self.tanh = nn.Tanh()


    def forward(self, x, info={}):
        x = self.tanh(self.affine1(x))
        h = self.tanh(sum([self.affine2(x), x]))
        a = F.log_softmax(self.head(h), dim=-1)
        v = self.value_head(h)

        return a, v


class Attention(nn.Module):
    def __init__(self, agent_config):
        super(Attention, self).__init__()
        self.args = argparse.Namespace(**agent_config)
        self.att_head = self.args.att_head
        self.hid_size = self.args.hid_size
        self.obs_shape = self.args.obs_shape
        self.n_actions = self.args.n_actions

        self.affine1 = nn.Linear(self.obs_shape, self.hid_size)
        self.affine2 = nn.Linear(self.hid_size, self.hid_size)
        self.attn = nn.MultiheadAttention(self.hid_size, self.att_head, batch_first=True)
        self.head = nn.Linear(self.hid_size,self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)
        self.tanh = nn.Tanh()

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_size, nhead=4, dim_feedforward=self.hid_size,
                                                         batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)




    def forward(self, x, info={}):
        x = self.tanh(self.affine1(x))
        h, _ = self.attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        #h = self.transformer(x.unsqueeze(0))
        #y = self.tanh(self.affine2(h))
        y = self.tanh(self.affine2(sum([h.squeeze(0), x])))

        a = F.log_softmax(self.head(y), dim=-1)
        v = self.value_head(y)

        return a, v