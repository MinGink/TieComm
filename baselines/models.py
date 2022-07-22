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

        self.affine1 = nn.Linear(self.args.obs_shape, self.args.hid_size)
        self.affine2 = nn.Linear(self.args.hid_size, self.args.hid_size)
        self.head = nn.Linear(self.args.hid_size,self.args.n_actions)
        self.value_head = nn.Linear(self.args.hid_size, 1)
        self.tanh = nn.Tanh()




    def forward(self, x, info={}):
        x = self.tanh(self.affine1(x))
        #x = self.tanh(self.affine2(x))
        h = self.tanh(sum([self.affine2(x), x]))
        a = F.softmax(self.head(x), dim=-1)
        v = self.value_head(h)

        return a, v





# class Random(nn.Module):
#     def __init__(self, args, num_inputs):
#         super(Random, self).__init__()
#         self.naction_heads = args.naction_heads
#
#         # Just so that pytorch is happy
#         self.parameter = nn.Parameter(torch.randn(3))
#
#     def forward(self, x, info={}):
#
#         sizes = x.size()[:-1]
#
#         v = Variable(torch.rand(sizes + (1,)), requires_grad=True)
#         out = []
#
#         for o in self.naction_heads:
#             var = Variable(torch.randn(sizes + (o, )), requires_grad=True)
#             out.append(F.log_softmax(var, dim=-1))
#
#         return out, v

