import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class DeepONet(nn.Module):
    def __init__(self, b_dim,  t_dim):
        super(DeepONet, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim

        self.branch = nn.Sequential(
            nn.Linear(self.b_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.t_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
        )

        # self.b = Parameter(torch.zeros(1))

    def forward(self, x, l):

        x = self.branch(x)
        l = self.trunk(l)
        res = torch.einsum("bi,bi->b", x, l)
        res = res.unsqueeze(1)  # + self.b
        return res

class DeepONet2D(nn.Module):
    def __init__(self, b_dim, t_dim):
        super(DeepONet2D, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim

        self.branch = nn.Sequential(
            nn.Linear(self.b_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.t_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

    def forward(self, x, l):
        x = self.branch(x)
        l = self.trunk(l)
        res = torch.einsum("bi,bi->b", x, l)
        res = res.unsqueeze(1)
        return res
    

#2d setting

# class Reshape(nn.Module):
#     def __init__(self, *args):
#         super(Reshape, self).__init__()
#         self.shape = args
#     def forward(self, x):
#         return x.view((x.size(0),)+self.shape)

class Dim2_multi_DeepONet(nn.Module):
    def __init__(self,b_dim,t_dim):
        super(Dim2_multi_DeepONet, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim

        # self.branch1 = nn.Sequential(
        #     nn.Linear(self.b_dim, 65*65),
        #     Reshape(1,65,65),
        #     nn.Conv2d(1, 64, 5, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 5, stride=2),
        #     nn.ReLU(),
        #     Reshape(128*14*14),
        #     nn.Linear(128*14*14, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128)
        #     )
        # self.branch2 = nn.Sequential(
        #     nn.Linear(self.b_dim, 65*65),
        #     Reshape(1,65,65),
        #     nn.Conv2d(1, 64, 5, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 5, stride=2),
        #     nn.ReLU(),
        #     Reshape(128*14*14),
        #     nn.Linear(128*14*14, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128)
        #     )
        self.branch1 = nn.Sequential(
            nn.Linear(self.b_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.branch2 = nn.Sequential(
            nn.Linear(self.b_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.trunk1 = nn.Sequential(
            nn.Linear(self.t_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.trunk2 = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )


        # self.b = Parameter(torch.zeros(1))


    def forward(self, x, l, e1):
        x1 = self.branch1(x)
        l = self.trunk1(l)
        x2 = self.branch2(x)
        e1 = self.trunk2(e1)
        res =  res = torch.einsum("bi,bi->b", x1, l) + torch.einsum("bi,bi->b", x2, e1)
        res = res.unsqueeze(-1) #+ self.b
        return res

