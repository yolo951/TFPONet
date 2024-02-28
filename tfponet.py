import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class DeepONet(nn.Module):
    def __init__(self, b_dim, t_dim):
        super(DeepONet, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim

        self.branch = nn.Sequential(
            nn.Linear(self.b_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.t_dim, 128),
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


class NN(nn.Module):
    def __init__(self, dim):
        super(NN, self).__init__()
        self.dim = dim

        self.nn = nn.Sequential(
            nn.Linear(self.dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.nn(x)
        return x


class NN2D(nn.Module):
    def __init__(self, dim):
        super(NN2D, self).__init__()
        self.dim = dim

        self.nn = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.nn(x)
        return x


class TFPONet(nn.Module):
    def __init__(self, b_dim, t_dim):
        super(TFPONet, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim

        self.deeponet = DeepONet(self.b_dim, self.t_dim)
        self.nn1 = NN(self.t_dim)
        self.nn2 = NN(self.t_dim)

    def forward(self, x, l, e1, e2):
        out0 = self.deeponet(x, l)
        out1 = self.nn1(e1)
        out2 = self.nn2(e2)

        res = out0 + out1 + out2
        res = res.unsqueeze(1)
        return res


class TFPONet2D(nn.Module):
    def __init__(self, b_dim, t_dim):
        super(TFPONet2D, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim

        self.deeponet = DeepONet2D(self.b_dim, self.t_dim)
        self.nn1 = NN2D(1)
        self.nn2 = NN2D(1)

    def forward(self, x, l, e1, e2):
        out0 = self.deeponet(x, l)
        out1 = self.nn1(e1)
        out2 = self.nn2(e2)

        res = out0 + out1 + out2
        res = res.unsqueeze(-1)
        return res



