
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
learning_rate = 0.0002
epochs = 1000
step_size = 100
gamma = 0.6
alpha = 1

# f=generate_data_1d.generate(end=y_end, samples=1100)
# np.save('ex4_f.npy',f)
f = np.load('ex3/f.npy')
# u = np.load('/home/v-tingdu/code/ex3/u_ex2_discontinuous.npy')
# u1 = np.zeros((1100, 501), dtype=np.float64)
# u2 = np.zeros((1100, 501), dtype=np.float64)
# u1[:] = u[:, :501]
# u2[:] = u[:, 500:]
# u2[:, 0] += 1
# u1, u2 = discontinuous_tfpm(f)
# np.save('/home/v-tingdu/code/ex3/u1.npy', u1)
# np.save('/home/v-tingdu/code/ex3/u2.npy', u2)
u1 = np.load('ex3/u1.npy')
u2 = np.load('ex3/u2.npy')
# idx = np.arange(0, 1001)
# for i in range(1):
#     plt.plot(idx, f[i].flatten())
# plt.savefig('/home/v-tingdu/code/ex3/daishan_f.png')
# plt.figure()
# for i in range(1):
#     plt.plot(idx[:501], u1[i].flatten())
#     plt.plot(idx[500:], u2[i].flatten())
# plt.savefig('/home/v-tingdu/code/ex3/daishan_u')

u1 *= factor
u2 *= factor
N_max = f.shape[-1]

loss_history = dict()


NS = 513
mse_history = []
mse_data_history = []
mse_jump_history = []
mse_jump_deriv_history = []
print("N value : ", NS - 1)

dim = 1001  # test resolution, dim must be odd
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
f_train = f[:ntrain, :]
N = f_train.shape[0] * int((NS - 1) / 2)

gridx_1 = np.linspace(0, 0.5, int((NS + 1) / 2))
gridx_2 = np.linspace(0.5, 1, int((NS + 1) / 2))
# gridy_1 = np.array([quad(integrand_y, 0, gridx_1[i])[0] for i in range(int((NS + 1) / 2))])
# gridy_2 = np.array([quad(integrand_y, 0, gridx_2[i])[0] for i in range(int((NS + 1) / 2))])
grid_h_1 = np.linspace(0, 0.5, int((N_max+1)/2))
grid_h_2 = np.linspace(0.5, 1, int((N_max+1)/2))
u_train_1 = interpolate.interp1d(grid_h_1, u1[:ntrain, :])(gridx_1[:-1])
u_train_2 = interpolate.interp1d(grid_h_2, u2[:ntrain, :])(gridx_2[1:])

ab_1 = get_modify(gridx_1, q1(gridx_1))
ab_2 = get_modify(gridx_2, q2(gridx_2))
input_f = np.repeat(f_train, int((NS - 1) / 2), axis=0)
input_loc_1 = np.tile(gridx_1[:-1], f_train.shape[0]).reshape((N, 1))
input_loc_2 = np.tile(gridx_2[1:], f_train.shape[0]).reshape((N, 1))
ab_1 = (ab_1 - min_modify_1)/(max_modify_1 - min_modify_1)
ab_2 = (ab_2 - min_modify_2)/(max_modify_2 - min_modify_2)
input_modify_1 = np.tile(ab_1[:-1], (f_train.shape[0], 1))
input_modify_2 = np.tile(ab_2[1:], (f_train.shape[0], 1))
output_1 = u_train_1.reshape((N, 1))
output_2 = u_train_2.reshape((N, 1))
input_f = torch.Tensor(input_f).to(device)
input_loc_1 = torch.Tensor(input_loc_1).to(device)
input_loc_2 = torch.Tensor(input_loc_2).to(device)
input_modify_1 = torch.Tensor(input_modify_1).to(device)
input_modify_2 = torch.Tensor(input_modify_2).to(device)
output_1 = torch.Tensor(output_1).to(device)
output_2 = torch.Tensor(output_2).to(device)
disc_point = torch.tensor([0.5]).to(device)
disc_point_airy = torch.zeros((2, 2)).to(device)
disc_point_airy[0] = torch.tensor(ab_1[-1])
disc_point_airy[1] = torch.tensor(ab_2[0])
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc_1, input_loc_2,
                                                                          input_modify_1, input_modify_2, output_1, output_2),
                                           batch_size=256, shuffle=True)

model_1 = TFPONet(NS,  1).to(device)
model_2 = TFPONet(NS,  1).to(device)
optimizer_1 = Adam(model_1.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=step_size, gamma=gamma)
optimizer_2 = Adam(model_2.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=step_size, gamma=gamma)
start = default_timer()
for ep in range(epochs):
    model_1.train()
    model_2.train()
    t1 = default_timer()
    train_mse = 0
    train_mse_data = 0
    train_mse_jump = 0
    train_mse_jump_deriv = 0
    for x, l_1, l_2, e_1, e_2, y_1, y_2 in train_loader:
        point = torch.tile(disc_point, (x.shape[0], 1))
        point_airy_00 = torch.tile(disc_point_airy[0, 0], (x.shape[0], 1))
        point_airy_01 = torch.tile(disc_point_airy[0, 1], (x.shape[0], 1))
        point_airy_10 = torch.tile(disc_point_airy[1, 0], (x.shape[0], 1))
        point_airy_11 = torch.tile(disc_point_airy[1, 1], (x.shape[0], 1))

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        out_1 = model_1(x, l_1, e_1[:, 0].reshape((-1, 1)), e_1[:, 1].reshape((-1, 1)))
        out_2 = model_2(x, l_2, e_2[:, 0].reshape((-1, 1)), e_2[:, 1].reshape((-1, 1)))
        mse_1 = F.mse_loss(out_1.view(out_1.numel(), -1), y_1.view(y_1.numel(), -1), reduction='mean')
        mse_2 = F.mse_loss(out_2.view(out_2.numel(), -1), y_2.view(y_2.numel(), -1), reduction='mean')
        mse = mse_1 + mse_2
        mse_data = mse.item()
        point.requires_grad_(True)
        out_1 = model_1(x, point, point_airy_00, point_airy_01)
        out_2 = model_2(x, point, point_airy_10, point_airy_11)
        mse_jump = 0.01 * F.mse_loss(out_2 - factor, out_1, reduction='mean')
        mse += mse_jump
        grad_1 = torch.autograd.grad(out_1, point, grad_outputs=torch.ones_like(out_1), create_graph=False,
                            only_inputs=True, retain_graph=True)[0]
        grad_2 = torch.autograd.grad(out_2, point, grad_outputs=torch.ones_like(out_2), create_graph=False,
                                     only_inputs=True, retain_graph=True)[0]
        mse_jump_deriv = 0.001 * F.mse_loss(grad_2 - factor, grad_1, reduction='mean')
        mse += mse_jump_deriv
        mse.backward()
        optimizer_1.step()
        optimizer_2.step()
        train_mse += mse.item()
        train_mse_data += mse_data
        train_mse_jump += mse_jump.item()
        train_mse_jump_deriv += mse_jump_deriv.item()
    scheduler_1.step()
    scheduler_2.step()
    # train_mse /= ntrain
    train_mse /= len(train_loader)
    t2 = default_timer()
    mse_history.append(train_mse)
    mse_data_history.append(train_mse_data / len(train_loader))
    mse_jump_history.append(train_mse_jump / len(train_loader))
    mse_jump_deriv_history.append(train_mse_jump_deriv / len(train_loader))
    print('Epoch {:d}/{:d}, MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1))
    # print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1), end='',
    #       flush=False)

print('Total training time:', default_timer() - start, 's')
loss_history["{}".format(NS)] = mse_history

torch.save(model_1.state_dict(), 'ex3/model1_256.pt')
torch.save(model_2.state_dict(), 'ex3/model2_256.pt')

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


pred_1 = pred_1.cpu()
pred_2 = pred_2.cpu()
residual_1 = pred_1 - u_test_1
residual_2 = pred_2 - u_test_2
fig = plt.figure(figsize=(8, 6), dpi=150)

plt.subplot(2, 2, 1)
plt.title("ground truth")
for i in range(ntest):
    plt.plot(grid_tx_1, u_test_1[i] / factor)
    plt.plot(grid_tx_2, u_test_2[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_g$:ground truth")
plt.grid()

plt.subplot(2, 2, 2)
plt.title("prediction")
for i in range(ntest):
    plt.plot(grid_tx_1, pred_1[i] / factor)
    plt.plot(grid_tx_2, pred_2[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_p$:predictions")
plt.grid()

plt.subplot(2, 2, 3)
plt.title("residuals")
for i in range(ntest):
    plt.plot(grid_tx_1, residual_1[i] / factor)
    plt.plot(grid_tx_2, residual_2[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_p$-$u_g$:residual")
plt.grid()

plt.subplot(2, 2, 4)
plt.title("training loss")
plt.plot(loss_history["{}".format(NS)], label='total loss')
plt.plot(mse_jump_history, label='jump loss')
plt.plot(mse_jump_deriv_history, label='jump deriv loss')
plt.plot(mse_data_history, label='data loss')
plt.yscale("log")
plt.xlabel("epoch")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show(block=True)
plt.savefig('ex3/loss_256.png')
