
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
# import generate_data_1d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



ntrain = 1000
ntest = 100
factor = 10
learning_rate = 0.0002
epochs = 500
step_size = 100
gamma = 0.6
alpha = 1

# f=generate_data_1d.generate(end=x_end, samples=1100)
# np.save('ex3/f.npy',f)
f = np.load('ex3/f.npy')
# u = discontinuous_tfpm(f)
# np.save('ex3/u.npy', u)
u1 = np.load('ex3/u1.npy')
u2 = np.load('ex3/u2.npy')

u1 *= factor
u2 *= factor
N_max = f.shape[-1]

loss_history = dict()


NS = 129
mse_history = []
mse_jump_history = []
mse_jump_deriv_history = []
print("N value : ", NS - 1)

f = interpolate.interp1d(np.linspace(0, 1, N_max), f)(np.linspace(0, 1, NS))
f_train = f[:ntrain, :]
N = f_train.shape[0] * int((NS - 1) / 2)

gridx_1 = np.linspace(0, 0.5 - 1/(NS - 1), int((NS - 1) / 2))
gridx_2 = np.linspace(0.5 + 1/(NS - 1), 1, int((NS - 1) / 2))
grid_h_1 = np.linspace(0, 0.5, int((N_max+1)/2))
grid_h_2 = np.linspace(0.5, 1, int((N_max+1)/2))
u_train_1 = interpolate.interp1d(grid_h_1, u1[:ntrain, :])(gridx_1)
u_train_2 = interpolate.interp1d(grid_h_2, u2[:ntrain, :])(gridx_2)

input_f = np.repeat(f_train, int((NS - 1) / 2), axis=0)
input_loc_1 = np.tile(gridx_1, f_train.shape[0]).reshape((N, 1))
input_loc_2 = np.tile(gridx_2, f_train.shape[0]).reshape((N, 1))

output_1 = u_train_1.reshape((N, 1))
output_2 = u_train_2.reshape((N, 1))
input_f = torch.Tensor(input_f).to(device)
input_loc_1 = torch.Tensor(input_loc_1).to(device)
input_loc_2 = torch.Tensor(input_loc_2).to(device)
output_1 = torch.Tensor(output_1).to(device)
output_2 = torch.Tensor(output_2).to(device)
disc_point = torch.tensor([0.5]).to(device)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc_1, input_loc_2, output_1, output_2),
                                           batch_size=256, shuffle=True)

model_1 = DeepONet(NS,  1).to(device)
model_2 = DeepONet(NS,  1).to(device)
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
    train_mse_jump = 0
    train_mse_jump_deriv = 0
    for x, l_1, l_2, y_1, y_2 in train_loader:
        point = torch.tile(disc_point, (x.shape[0], 1))
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        out_1 = model_1(x, l_1)
        out_2 = model_2(x, l_2)
        mse_1 = F.mse_loss(out_1.view(out_1.numel(), -1), y_1.view(y_1.numel(), -1), reduction='mean')
        mse_2 = F.mse_loss(out_2.view(out_2.numel(), -1), y_2.view(y_2.numel(), -1), reduction='mean')
        mse = mse_1 + mse_2
        point.requires_grad_(True)
        out_1 = model_1(x, point)
        out_2 = model_2(x, point)
        mse_jump = F.mse_loss(out_2 - factor, out_1, reduction='mean')
        mse += 0.01 * mse_jump
        grad_1 = torch.autograd.grad(out_1, point, grad_outputs=torch.ones_like(out_1), create_graph=False,
                            only_inputs=True, retain_graph=True)[0]
        grad_2 = torch.autograd.grad(out_2, point, grad_outputs=torch.ones_like(out_2), create_graph=False,
                                     only_inputs=True, retain_graph=True)[0]
        mse_jump_deriv = F.mse_loss(grad_2 - factor, grad_1, reduction='mean')
        mse += 0.001 * mse_jump_deriv
        mse.backward()
        optimizer_1.step()
        optimizer_2.step()
        train_mse += mse.item()
        train_mse_jump += mse_jump.item()
        train_mse_jump_deriv += mse_jump_deriv.item()
    scheduler_1.step()
    scheduler_2.step()
    # train_mse /= ntrain
    train_mse /= len(train_loader)
    t2 = default_timer()
    mse_history.append(train_mse)
    mse_jump_history.append(train_mse_jump / len(train_loader))
    mse_jump_deriv_history.append(train_mse_jump_deriv / len(train_loader))
    print('Epoch {:d}/{:d}, MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1))
    # print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1), end='',
    #       flush=False)

print('Total training time:', default_timer() - start, 's')
loss_history["{}".format(NS)] = mse_history

torch.save(model_1.state_dict(), 'ex3/vanilla_model1_128.pt')
torch.save(model_2.state_dict(), 'ex3/vanilla_model2_128.pt')

dim = 513  # test resolution, dim must be odd
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


pred_1 = pred_1.cpu()
pred_2 = pred_2.cpu()
residual_1 = pred_1 - u_test_1
residual_2 = pred_2 - u_test_2
fig = plt.figure(figsize=(8, 6), dpi=150)

plt.subplot(2, 2, 1)
plt.title("ground truth")
for i in range(ntest):
    plt.plot(grid_t_1, u_test_1[i] / factor)
    plt.plot(grid_t_2, u_test_2[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_g$:ground truth")
plt.grid()

plt.subplot(2, 2, 2)
plt.title("prediction")
for i in range(ntest):
    plt.plot(grid_t_1, pred_1[i] / factor)
    plt.plot(grid_t_2, pred_2[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_p$:predictions")
plt.grid()

plt.subplot(2, 2, 3)
plt.title("residuals")
for i in range(ntest):
    plt.plot(grid_t_1, residual_1[i] / factor)
    plt.plot(grid_t_2, residual_2[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_p$-$u_g$:residual")
plt.grid()

plt.subplot(2, 2, 4)
plt.title("training loss")
plt.plot(loss_history["{}".format(NS)], label='total loss')
plt.plot(mse_jump_history, label='jump loss')
plt.plot(mse_jump_deriv_history, label='jump deriv loss')
plt.yscale("log")
plt.xlabel("epoch")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show(block=True)
plt.savefig('ex3/vanilla_loss_128.png')
