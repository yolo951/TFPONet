
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
from tfponet import TFPONet
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


def q(x):
    return np.where(x<=0.5, 5000.0, 100.0*(4.0+32.0*x))

def q1(x):
    return 5000.0*np.ones_like(x)

def q2(x):
    return 100.0*(4.0+32.0*x)

def get_coeff(grid, f):
    N = len(grid)
    coefficients = np.zeros((N, 2))
    for i in range(N - 1):
        a = (f[i + 1] - f[i]) / (grid[i + 1] - grid[i])
        b = f[i] - a * grid[i]
        coefficients[i][0] = a
        coefficients[i][1] = b
    coefficients[int(N/2)-1] = coefficients[int(N/2)-2]
    coefficients[-1, :] = coefficients[-2]
    return coefficients

def get_modify(x, f):
    coeff = get_coeff(x, f)
    ab = np.zeros((len(x), 2), dtype=np.float64)
    for i in range(len(x)):
        if abs(coeff[i][0]) >= 1e-8:
            ab[i, 0], aip, ab[i, 1], bip = airy(f[i] * np.power(np.abs(coeff[i][0]), -2 / 3))
        else:
            if coeff[i][1] >= 1e-8:
                ab[i, 1] = np.exp(x[i] * sqrt(coeff[i][1]))
                ab[i, 0] = np.exp(-x[i] * sqrt(coeff[i][1]))
            else:
                ab[i, 1], ab[i, 0] = 1.0, x[i]
    return ab


ntrain = 1000
ntest = 100
factor = 10
learning_rate = 0.0005
epochs = 1000
step_size = 60
gamma = 0.6
alpha = 1
NS = 129
dim = 1001  # test resolution, dim must be odd


f = np.load('ex3/f.npy')
u1 = np.load('ex3/u1.npy')
u2 = np.load('ex3/u2.npy')
model_1 = TFPONet(NS,  1).to(device)
model_2 = TFPONet(NS,  1).to(device)
model_1.load_state_dict(torch.load('ex3/model1_128.pt'))
model_2.load_state_dict(torch.load('ex3/model2_128.pt'))
u1 *= factor
u2 *= factor
N_max = f.shape[-1]



grid_tx_1 = np.linspace(0, 0.5, int((dim + 1) / 2))
grid_tx_2 = np.linspace(0.5, 1, int((dim + 1) / 2))
# grid_ty_1 = np.array([quad(integrand_y, 0, grid_tx_1[i])[0] for i in range(int((dim + 1) / 2))])
# grid_ty_2 = np.array([quad(integrand_y, 0, grid_tx_2[i])[0] for i in range(int((dim + 1) / 2))])
ab_1 = get_modify(grid_tx_1, q1(grid_tx_1))
ab_2 = get_modify(grid_tx_2, q2(grid_tx_2))
max_modify_1 = ab_1.max(axis=0)
min_modify_1 = ab_1.min(axis=0)
max_modify_2 = ab_2.max(axis=0)
min_modify_2 = ab_2.min(axis=0)

f = interpolate.interp1d(np.linspace(0, 1, N_max), f)(np.linspace(0, 1, NS))
grid_h_1 = np.linspace(0, 0.5, int((N_max+1)/2))
grid_h_2 = np.linspace(0.5, 1, int((N_max+1)/2))



batch_size = int((dim- 1) / 2)
N = ntest * int((dim- 1) / 2)
grid_tx_1 = np.linspace(0, 0.5 - 1/(dim - 1), int((dim - 1) / 2))
grid_tx_2 = np.linspace(0.5 + 1/(dim - 1), 1, int((dim - 1) / 2))
# grid_ty_1 = np.array([quad(integrand_y, 0, grid_tx_1[i])[0] for i in range(int((dim - 1) / 2))])
# grid_ty_2 = np.array([quad(integrand_y, 0, grid_tx_2[i])[0] for i in range(int((dim - 1) / 2))])
f_test = f[-ntest:, :]
u_test_1 = interpolate.interp1d(grid_h_1, u1[-ntest:, :])(grid_tx_1)
u_test_2 = interpolate.interp1d(grid_h_2, u2[-ntest:, :])(grid_tx_2)

ab_1 = get_modify(grid_tx_1, q1(grid_tx_1))
ab_2 = get_modify(grid_tx_2, q2(grid_tx_2))
input_f = np.repeat(f_test, int((dim - 1) / 2), axis=0)
input_loc_1 = np.tile(grid_tx_1, f_test.shape[0]).reshape((N, 1))
input_loc_2 = np.tile(grid_tx_2, f_test.shape[0]).reshape((N, 1))
input_modify_1 = np.tile(ab_1, (f_test.shape[0], 1))
input_modify_2 = np.tile(ab_2, (f_test.shape[0], 1))
output_1 = u_test_1.reshape((N, 1))
output_2 = u_test_2.reshape((N, 1))

input_modify_1 = (input_modify_1 - min_modify_1)/(max_modify_1 - min_modify_1)
input_modify_2 = (input_modify_2 - min_modify_2)/(max_modify_2 - min_modify_2)
input_f = torch.Tensor(input_f).to(device)
input_loc_1 = torch.Tensor(input_loc_1).to(device)
input_loc_2 = torch.Tensor(input_loc_2).to(device)
input_modify_1 = torch.Tensor(input_modify_1).to(device)
input_modify_2 = torch.Tensor(input_modify_2).to(device)
output_1 = torch.Tensor(output_1).to(device)
output_2 = torch.Tensor(output_2).to(device)
test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc_1, input_loc_2,
                                                                          input_modify_1, input_modify_2, output_1, output_2),
                                            batch_size=batch_size, shuffle=False)

pred_1 = torch.zeros((ntest, int((dim- 1) / 2)))
pred_2 = torch.zeros((ntest, int((dim- 1) / 2)))
index = 0
test_mse = 0
with torch.no_grad():
    for x, l_1, l_2, e_1, e_2, y_1, y_2 in test_h_loader:
        out_1 = model_1(x, l_1, e_1[:, 0].reshape((-1, 1)), e_1[:, 1].reshape((-1, 1)))
        out_2 = model_2(x, l_2, e_2[:, 0].reshape((-1, 1)), e_2[:, 1].reshape((-1, 1)))
        pred_1[index] = out_1.view(-1)
        pred_2[index] = out_2.view(-1)
        mse_1 = F.mse_loss(out_1.view(1, -1), y_1.view(1, -1), reduction='mean')
        mse_2 = F.mse_loss(out_2.view(1, -1), y_2.view(1, -1), reduction='mean')
        test_mse += mse_1.item() + mse_2.item()
        index += 1
    test_mse /= len(test_h_loader)
    print('test error on high resolution: MSE = ', test_mse)

# pred_1 = pred_1.cpu()
# pred_2 = pred_2.cpu()
# residual_1 = pred_1 - u_test_1
# residual_2 = pred_2 - u_test_2
# for i in range(1):
#     fig = plt.figure()
#     plt.plot(grid_tx_1, pred_1[i] / factor, linestyle='solid', linewidth=1.5, color='blue', label='ground truth')
#     plt.plot(grid_tx_2, pred_2[i] / factor, linestyle='solid', linewidth=1.5, color='blue')
#     plt.plot(grid_tx_1, u_test_1[i] / factor, linestyle='--', linewidth=1.5, color='red', label='prediction')
#     plt.plot(grid_tx_2, u_test_2[i] / factor, linestyle='--', linewidth=1.5, color='red')
#     # plt.show()
#     plt.legend()
#     plt.savefig('ex3/testfig/fig{}.eps'.format(i))

