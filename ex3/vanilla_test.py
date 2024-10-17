
import sys
sys.path.append('D:\pycharm\pycharm_project\TFPONet')
import pdb

import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from timeit import default_timer
from Adam import Adam

import tfponet
importlib.reload(tfponet)
from deeponet import DeepONet
from scipy import interpolate
from scipy.special import airy
from scipy.integrate import quad
# from tfpm import discontinuous_tfpm
from math import sqrt
# import generate_data_1d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def f_a(x):
#     return np.where(x <= 0.5, 1 / (2 + 4 * np.sin(x) ** 2), 0.5)


# integrand_y = lambda x: 1 / f_a(x)
# # 原始问题在0.5处间断
# y_disc = quad(integrand_y, 0, 0.5)[0]
# y_end = quad(integrand_y, 0, 1)[0]



ntrain = 1000
ntest = 100
factor = 10
learning_rate = 0.0005
epochs = 1000
step_size = 60
gamma = 0.6
alpha = 1
NS = 129
dim = 2049  # test resolution, dim must be odd


f = np.load('/home/v-tingdu/code/ex3/f.npy')
u1 = np.load('/home/v-tingdu/code/ex3/u1.npy')
u2 = np.load('/home/v-tingdu/code/ex3/u2.npy')
model_1 = DeepONet(NS,  1).to(device)
model_2 = DeepONet(NS,  1).to(device)
model_1.load_state_dict(torch.load('/home/v-tingdu/code/ex3/vanilla_model1_128.pt'))
model_2.load_state_dict(torch.load('/home/v-tingdu/code/ex3/vanilla_model2_128.pt'))
u1 *= factor
u2 *= factor
N_max = f.shape[-1]



f = interpolate.interp1d(np.linspace(0, 1, N_max), f)(np.linspace(0, 1, NS))
grid_h_1 = np.linspace(0, 0.5, int((N_max+1)/2))
grid_h_2 = np.linspace(0.5, 1, int((N_max+1)/2))
batch_size = int((dim- 1) / 2)
N = ntest * int((dim- 1) / 2)
grid_t_1 = np.linspace(0, 0.5 - 1/(dim - 1), int((dim - 1) / 2))
grid_t_2 = np.linspace(0.5 + 1/(dim - 1), 1, int((dim - 1) / 2))
f_test = f[-ntest:, :]
u_test_1 = interpolate.interp1d(grid_h_1, u1[-ntest:, :])(grid_t_1)
u_test_2 = interpolate.interp1d(grid_h_2, u2[-ntest:, :])(grid_t_2)
input_f = np.repeat(f_test, int((dim - 1) / 2), axis=0)
input_loc_1 = np.tile(grid_t_1, f_test.shape[0]).reshape((N, 1))
input_loc_2 = np.tile(grid_t_2, f_test.shape[0]).reshape((N, 1))
output_1 = u_test_1.reshape((N, 1))
output_2 = u_test_2.reshape((N, 1))
input_f = torch.Tensor(input_f).to(device)
input_loc_1 = torch.Tensor(input_loc_1).to(device)
input_loc_2 = torch.Tensor(input_loc_2).to(device)
output_1 = torch.Tensor(output_1).to(device)
output_2 = torch.Tensor(output_2).to(device)
test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc_1, input_loc_2, output_1, output_2),
                                            batch_size=batch_size, shuffle=False)

pred_1 = torch.zeros((ntest, int((dim- 1) / 2)))
pred_2 = torch.zeros((ntest, int((dim- 1) / 2)))
index = 0
test_mse = 0
with torch.no_grad():
    for x, l_1, l_2, y_1, y_2 in test_h_loader:
        out_1 = model_1(x, l_1)
        out_2 = model_2(x, l_2)
        pred_1[index] = out_1.view(-1)
        pred_2[index] = out_2.view(-1)
        mse_1 = F.mse_loss(out_1.view(1, -1), y_1.view(1, -1), reduction='mean')
        mse_2 = F.mse_loss(out_2.view(1, -1), y_2.view(1, -1), reduction='mean')
        test_mse += mse_1.item() + mse_2.item()
        index += 1
    test_mse /= len(test_h_loader)
    print('test error on high resolution: MSE = ', test_mse)



