import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.affine1 = nn.Linear(args.obs_shape, args.hid_size)
        self.affine2 = nn.Linear(args.hid_size, args.hid_size)
        self.head = nn.Linear(args.hid_size,args.n_actions)
        self.value_head = nn.Linear(args.hid_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, info={}):
        x = self.tanh(self.affine1(x))
        h = self.tanh(sum([self.affine2(x), x]))
        v = self.value_head(h)
        a = F.softmax(self.head(h), dim=-1)

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

