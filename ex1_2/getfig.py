
import sys
sys.path.append('D:\pycharm\pycharm_project\TFPONet')

import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from timeit import default_timer
from Adam import Adam
from tfponet import TFPONet
from scipy import interpolate
from scipy.special import airy
# from ex1_tfpm import tfpm
import generate_data_1d as generate_data_1d
from math import sqrt

# importlib.reload(generate_data_1d)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def f_a(x):
    return np.where(x <= 0.5, 1, 0.0001)


def q(x):
    return np.where(x <= 0.5, 2 * x + 1, 2 * (1 - x) + 1) / f_a(x)


def q_1(x):
    return 2 * x + 1


def q2(x):
    return (2 * (1 - x) + 1) / 0.0001


def get_coeff(grid, f):
    N = len(grid)
    coefficients = np.zeros((N, 2))
    for i in range(N - 1):
        a = (f[i + 1] - f[i]) / (grid[i + 1] - grid[i])
        b = f[i] - a * grid[i]
        coefficients[i][0] = a
        coefficients[i][1] = b
    coefficients[int(N / 2) - 1] = coefficients[int(N / 2) - 2]
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


eps = 0.001
NS = 129
ntrain = 1000
ntest = 100
ntest_show = 100
factor = 10
learning_rate = 0.001
epochs = 1000
step_size = 50
gamma = 0.6
alpha = 1

f = np.load('ex1_2/f.npy')
u = np.load('ex1_2/u.npy')
u *= factor
f[:, 640:] /= 0.0001

N_max = f.shape[-1]
model = TFPONet(NS,  1).to(device)
model.load_state_dict(torch.load('ex1_2/model_128.pt'))




loss_history = dict()
mse_history = []
print("N value : ", NS - 1)


N = ntrain * NS
grid_h = np.linspace(0, 1, N_max)
gridx = np.linspace(0, 1, NS)
ab = get_modify(gridx, q(gridx))
f_train = f[:ntrain, :]
f_train = interpolate.interp1d(grid_h, f_train)(gridx)
input_f = np.repeat(f_train, NS, axis=0)
max_f = input_f.max(axis=0)
min_f = input_f.min(axis=0)
max_modify = ab.max(axis=0)
min_modify = ab.min(axis=0)


dim = 257 # test resolution, dim must be odd
batch_size = dim
N = ntest * dim
f_test = f[-ntest:, :]
grid_t = np.linspace(0, 1, dim)
u_test = u[-ntest:, :]
f_test = interpolate.interp1d(grid_h, f_test)(gridx)
u_test = interpolate.interp1d(grid_h, u_test)(grid_t)
ab = get_modify(grid_t, q(grid_t))
input_f = np.repeat(f_test, dim, axis=0)
input_loc = np.tile(grid_t, ntest).reshape((N, 1))
input_modify = np.tile(ab, (ntest, 1))
output = u_test.reshape((N, 1))
input_modify = (input_modify - min_modify)/(max_modify - min_modify)
input_f = (input_f - min_f)/(max_f - min_f)
input_f = torch.Tensor(input_f).to(device)
input_loc = torch.Tensor(input_loc).to(device)
input_modify = torch.Tensor(input_modify).to(device)
output = torch.Tensor(output).to(device)
test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc,
                                                                          input_modify, output),
                                            batch_size=batch_size, shuffle=False)

pred = torch.zeros((ntest, dim))
index = 0
test_mse = 0
with torch.no_grad():
    for x, l, e, y in test_h_loader:
        out = model(x, l, e[:, 0].reshape((-1, 1)), e[:, 1].reshape((-1, 1)))
        pred[index] = out.view(-1)
        mse = F.mse_loss(out.view(1, -1), y.view(1, -1), reduction='mean')
        test_mse += mse.item()
        index += 1
    test_mse /= len(test_h_loader)
    print('test error on high resolution: MSE = ', test_mse)


pred = pred.cpu()
residual = pred - u_test

i = 0
fig = plt.figure()
plt.plot(grid_t, u_test[i] / factor, label='ground truth',linestyle='solid', linewidth=1.5, color='blue')
plt.plot(grid_t, pred[i] / factor, label='prediction', linestyle='--', linewidth=1.5, color='red')
plt.xlabel("x", fontsize=12)
plt.ylabel("$u$", fontsize=12)
legend2 = plt.legend(fontsize=12, frameon=True)
ax2= plt.gca()
ax2.tick_params(axis='both', which='major', labelsize=10)
plt.show()

fig = plt.figure(figsize=(6.4,2))#默认figsize=(6.4,4.8)
plt.plot(grid_t[int((dim+1)/2)-10:int((dim+1)/2)+10], u_test[i, int((dim+1)/2)-10:int((dim+1)/2)+10] / factor,
         linestyle='solid', linewidth=1.5, color='blue', label='ground truth',)
plt.plot(grid_t[int((dim+1)/2)-10:int((dim+1)/2)+10],pred[i, int((dim+1)/2)-10:int((dim+1)/2)+10] / factor,
         linestyle='--', linewidth=1.5, color='red', label='prediction')
legend2 = plt.legend(fontsize=10, frameon=True)
ax2= plt.gca()
ax2.tick_params(axis='both', which='major', labelsize=10)
plt.show()


fig = plt.figure(figsize=(6.4,2))#默认figsize=(6.4,4.8)
plt.plot(grid_t[-15:], u_test[i, -15:] / factor,
         linestyle='solid', linewidth=1.5, color='blue', label='ground truth',)
plt.plot(grid_t[-15:], pred[i, -15:] / factor,
         linestyle='--', linewidth=1.5, color='red', label='prediction')
legend2 = plt.legend(fontsize=10, frameon=True)
ax2= plt.gca()
ax2.tick_params(axis='both', which='major', labelsize=10)
plt.show()


# for i in range(ntest):
#     fig = plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     # plt.title("ground truth")
#     plt.plot(grid_t, u_test[i] / factor, label='ground truth',linestyle='solid', linewidth=1.5, color='blue')
#     plt.plot(grid_t, pred[i] / factor, label='prediction', linestyle='--', linewidth=1.5, color='red')
#     plt.xlabel("x", fontsize=12)
#     plt.ylabel("$u$", fontsize=12)
#     legend2 = plt.legend(fontsize=12, frameon=True)
#     ax2= plt.gca()
#     ax2.tick_params(axis='both', which='major', labelsize=10)
#     # plt.show()
#
#     plt.subplot(1, 3, 2)
#     # fig = plt.figure(figsize=(6.4,2))#默认figsize=(6.4,4.8)
#     plt.plot(grid_t[int((dim+1)/2)-10:int((dim+1)/2)+10], u_test[i, int((dim+1)/2)-10:int((dim+1)/2)+10] / factor,
#              linestyle='solid', linewidth=1.5, color='blue', label='ground truth',)
#     plt.plot(grid_t[int((dim+1)/2)-10:int((dim+1)/2)+10],pred[i, int((dim+1)/2)-10:int((dim+1)/2)+10] / factor,
#              linestyle='--', linewidth=1.5, color='red', label='prediction')
#     legend2 = plt.legend(fontsize=10, frameon=True)
#     ax2= plt.gca()
#     ax2.tick_params(axis='both', which='major', labelsize=10)
#     # plt.show()
#
#     plt.subplot(1, 3, 3)
#     # fig = plt.figure(figsize=(6.4,2))#默认figsize=(6.4,4.8)
#     plt.plot(grid_t[-15:], u_test[i, -15:] / factor,
#              linestyle='solid', linewidth=1.5, color='blue', label='ground truth',)
#     plt.plot(grid_t[-15:], pred[i, -15:] / factor,
#              linestyle='--', linewidth=1.5, color='red', label='prediction')
#     legend2 = plt.legend(fontsize=10, frameon=True)
#     ax2= plt.gca()
#     ax2.tick_params(axis='both', which='major', labelsize=10)
#     plt.savefig('ex1_2/alltestfig/ex1_fig{}.png'.format(i))

