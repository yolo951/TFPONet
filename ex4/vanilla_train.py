

import sys
sys.path.append('/home/v-tingdu/code')
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from timeit import default_timer
from Adam import Adam
from deeponet import DeepONet
from scipy import interpolate
from scipy.special import airy
# from discontinuous_tfpm import discontinuous_tfpm
from math import sqrt
from scipy.integrate import quad


# importlib.reload(generate_data_1d)
# importlib.reload(deeponet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



eps = 0.001
ntrain = 1000
ntest = 100
factor = 10
learning_rate = 0.0002
epochs = 1000
step_size = 100
gamma = 0.6
alpha = 1


# f=generate_data_1d.generate(samples=1100)
# np.save('f_multi_distribution.npy',f)
f = np.load('ex4/f.npy')
# u=discontinuous_tfpm(f)
# np.save('u_ex2_discontinuous.npy', u)
u1 = np.load('ex4/u1.npy')
u2 = np.load('ex4/u2.npy')
u3 = np.load('ex4/u3.npy')


u1 *= factor
u2 *= factor
u3 *= factor
N_max = f.shape[-1]


loss_history = dict()


NS = 302
mse_history = []
mse_jump_history = []
mse_jump_deriv_history = []
mse_data_history = []
print("N value : ", NS - 1)

f = interpolate.interp1d(np.linspace(0, 1, N_max), f)(np.linspace(0, 1, NS))
f_train = f[:ntrain, :]
N = f_train.shape[0] * int((NS - 2) / 3)


gridx_1 = np.linspace(0, 0.25 - 3/(NS - 2)/4, int((NS - 2) / 3))
gridx_2 = np.linspace(0.25 + 3/(NS + 1)/4, 0.5 - 3/(NS + 1)/4, int((NS - 2) / 3))
gridx_3 = np.linspace(0.5 + 3/(NS - 2)/2, 1, int((NS - 2) / 3))

gridx_h = np.linspace(0, 1, N_max)
grid_h_1 = gridx_h[:int((N_max-1)/4+1)]
grid_h_2 = gridx_h[int((N_max-1)/4):int((N_max-1)/2+1)]
grid_h_3 = gridx_h[int((N_max-1)/2):]
u_train_1 = interpolate.interp1d(grid_h_1, u1[:ntrain, :])(gridx_1)
u_train_2 = interpolate.interp1d(grid_h_2, u2[:ntrain, :])(gridx_2)
u_train_3 = interpolate.interp1d(grid_h_3, u3[:ntrain, :])(gridx_3)
input_f = np.repeat(f_train, int((NS - 1) / 3), axis=0)
input_loc_1 = np.tile(gridx_1, ntrain).reshape((N, 1))
input_loc_2 = np.tile(gridx_2, ntrain).reshape((N, 1))
input_loc_3 = np.tile(gridx_3, ntrain).reshape((N, 1))
output_1 = u_train_1.reshape((N, 1))
output_2 = u_train_2.reshape((N, 1))
output_3 = u_train_3.reshape((N, 1))
input_f = torch.Tensor(input_f).to(device)
input_loc_1 = torch.Tensor(input_loc_1).to(device)
input_loc_2 = torch.Tensor(input_loc_2).to(device)
input_loc_3 = torch.Tensor(input_loc_3).to(device)
output_1 = torch.Tensor(output_1).to(device)
output_2 = torch.Tensor(output_2).to(device)
output_3 = torch.Tensor(output_3).to(device)
disc_point_1 = torch.tensor([0.25]).to(device)
disc_point_2 = torch.tensor([0.5]).to(device)
disc_point_3 = torch.tensor([1.0]).to(device)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc_1, input_loc_2, input_loc_3,
                                                                          output_1, output_2, output_3),
                                                                          batch_size=256, shuffle=True)

model_1 = DeepONet(NS,  1).to(device)
model_2 = DeepONet(NS,  1).to(device)
model_3 = DeepONet(NS,  1).to(device)
optimizer_1 = Adam(model_1.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=step_size, gamma=gamma)
optimizer_2 = Adam(model_2.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=step_size, gamma=gamma)
optimizer_3 = Adam(model_3.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler_3 = torch.optim.lr_scheduler.StepLR(optimizer_3, step_size=step_size, gamma=gamma)
start = default_timer()
for ep in range(epochs):
    model_1.train()
    model_2.train()
    t1 = default_timer()
    train_mse = 0
    train_mse_jump = 0
    train_mse_jump_deriv = 0
    train_data_mse = 0
    for x, l_1, l_2, l_3, y_1, y_2, y_3 in train_loader:
        point_1 = torch.tile(disc_point_1, (x.shape[0], 1))
        point_2 = torch.tile(disc_point_2, (x.shape[0], 1))
        point_3 = torch.tile(disc_point_3, (x.shape[0], 1))
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_3.zero_grad()
        out_1 = model_1(x, l_1)
        out_2 = model_2(x, l_2)
        out_3 = model_3(x, l_3)
        mse_1 = F.mse_loss(out_1.view(out_1.numel(), -1), y_1.view(y_1.numel(), -1), reduction='mean')
        mse_2 = F.mse_loss(out_2.view(out_2.numel(), -1), y_2.view(y_2.numel(), -1), reduction='mean')
        mse_3 = F.mse_loss(out_3.view(out_3.numel(), -1), y_3.view(y_3.numel(), -1), reduction='mean')
        mse = mse_1 + mse_2 + mse_3
        point_1.requires_grad_(True)
        out_1 = model_1(x, point_1)
        out_2 = model_2(x, point_1)
        mse_jump_1 = 0.01*F.mse_loss(out_2 - (-1)*factor, out_1, reduction='mean')
        mse += mse_jump_1
        grad_1 = torch.autograd.grad(out_1, point_1, grad_outputs=torch.ones_like(out_1), create_graph=False,
                            only_inputs=True, retain_graph=True)[0]
        grad_2 = torch.autograd.grad(out_2, point_1, grad_outputs=torch.ones_like(out_2), create_graph=False,
                                     only_inputs=True, retain_graph=True)[0]
        mse_jump_deriv_1 = 0.0001 * F.mse_loss(grad_2 - 1*factor, grad_1, reduction='mean')
        mse += mse_jump_deriv_1
        point_2.requires_grad_(True)
        out_2 = model_2(x, point_2)
        out_3 = model_3(x, point_2)
        mse_jump_2 = 0.01*F.mse_loss(out_3 - 1*factor, out_2, reduction='mean')
        mse += mse_jump_2
        grad_2 = torch.autograd.grad(out_2, point_2, grad_outputs=torch.ones_like(out_2), create_graph=False,
                                     only_inputs=True, retain_graph=True)[0]
        grad_3 = torch.autograd.grad(out_3, point_2, grad_outputs=torch.ones_like(out_3), create_graph=False,
                                     only_inputs=True, retain_graph=True)[0]
        mse_jump_deriv_2 = 0.0001 * F.mse_loss(grad_3 - 0*factor, grad_2, reduction='mean')
        mse += mse_jump_deriv_2

        out_data_end = model_3(x, point_3)
        mse_data_end = 10 * F.mse_loss(out_data_end, 0*out_data_end, reduction='mean')
        mse += mse_data_end
        mse.backward()
        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()
        train_mse += mse.item()
        train_mse_jump += mse_jump_1.item() + mse_jump_2.item()
        train_mse_jump_deriv += mse_jump_deriv_1.item()
        train_data_mse += mse_1.item() + mse_2.item() + mse_3.item() + mse_data_end.item()
    scheduler_1.step()
    scheduler_2.step()
    # train_mse /= ntrain
    train_mse /= len(train_loader)
    t2 = default_timer()
    mse_history.append(train_mse)
    mse_jump_history.append(train_mse_jump / len(train_loader))
    mse_jump_deriv_history.append(train_mse_jump_deriv / len(train_loader))
    mse_data_history.append(train_data_mse / len(train_loader))
    print('Epoch {:d}/{:d}, MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1))
    # print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1), end='',
    #       flush=False)

print('Total training time:', default_timer() - start, 's')
loss_history["{}".format(NS)] = mse_history

torch.save(model_1.state_dict(), 'ex4/vanilla_model1_300.pt')
torch.save(model_2.state_dict(), 'ex4/vanilla_model2_300.pt')
torch.save(model_3.state_dict(), 'ex4/vanilla_model3_300.pt')

dim = 902  # test resolution, dim must be odd
batch_size = int((dim- 2) / 3)
N = ntest * int((dim- 2) / 3)
grid_tx_1 = np.linspace(0, 0.25 - 3/(dim - 2)/4, int((dim - 2) / 3))
grid_tx_2 = np.linspace(0.25 + 3/(dim + 1)/4, 0.5 - 3/(dim + 1)/4, int((dim - 2) / 3))
grid_tx_3 = np.linspace(0.5 + 3/(dim - 2)/2, 1, int((dim - 2) / 3))
f_test = f[-ntest:, :]
u_test_1 = interpolate.interp1d(grid_h_1, u1[-ntest:, :])(grid_tx_1)
u_test_2 = interpolate.interp1d(grid_h_2, u2[-ntest:, :])(grid_tx_2)
u_test_3 = interpolate.interp1d(grid_h_3, u3[-ntest:, :])(grid_tx_3)
input_f = np.repeat(f_test, int((dim - 1) / 3), axis=0)
input_loc_1 = np.tile(grid_tx_1, ntest).reshape((N, 1))
input_loc_2 = np.tile(grid_tx_2, ntest).reshape((N, 1))
input_loc_3 = np.tile(grid_tx_3, ntest).reshape((N, 1))
output_1 = u_test_1.reshape((N, 1))
output_2 = u_test_2.reshape((N, 1))
output_3 = u_test_3.reshape((N, 1))
input_f = torch.Tensor(input_f).to(device)
input_loc_1 = torch.Tensor(input_loc_1).to(device)
input_loc_2 = torch.Tensor(input_loc_2).to(device)
input_loc_3 = torch.Tensor(input_loc_3).to(device)
output_1 = torch.Tensor(output_1).to(device)
output_2 = torch.Tensor(output_2).to(device)
output_3 = torch.Tensor(output_3).to(device)
test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc_1, input_loc_2, input_loc_3,
                                                                           output_1, output_2, output_3),
                                                                           batch_size=batch_size, shuffle=False)

pred_1 = torch.zeros((ntest, int((dim- 1) / 3)))
pred_2 = torch.zeros((ntest, int((dim- 1) / 3)))
pred_3 = torch.zeros((ntest, int((dim- 1) / 3)))
index = 0
test_mse = 0
with torch.no_grad():
    for x, l_1, l_2, l_3, y_1, y_2, y_3 in test_h_loader:
        out_1 = model_1(x, l_1)
        out_2 = model_2(x, l_2)
        out_3 = model_3(x, l_3)
        pred_1[index] = out_1.view(-1)
        pred_2[index] = out_2.view(-1)
        pred_3[index] = out_3.view(-1)
        mse_1 = F.mse_loss(out_1.view(1, -1), y_1.view(1, -1), reduction='mean')
        mse_2 = F.mse_loss(out_2.view(1, -1), y_2.view(1, -1), reduction='mean')
        mse_3 = F.mse_loss(out_3.view(1, -1), y_3.view(1, -1), reduction='mean')
        test_mse += mse_1.item() + mse_2.item() + mse_3.item()
        index += 1
    test_mse /= len(test_h_loader)
    print('test error on high resolution: MSE = ', test_mse)


pred_1 = pred_1.cpu()
pred_2 = pred_2.cpu()
pred_3 = pred_3.cpu()
residual_1 = pred_1 - u_test_1
residual_2 = pred_2 - u_test_2
residual_3 = pred_3 - u_test_3
fig = plt.figure(figsize=(8, 6), dpi=150)

plt.subplot(2, 2, 1)
plt.title("ground truth")
for i in range(ntest):
    plt.plot(grid_tx_1, u_test_1[i] / factor)
    plt.plot(grid_tx_2, u_test_2[i] / factor)
    plt.plot(grid_tx_3, u_test_3[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_g$:ground truth")
plt.grid()

plt.subplot(2, 2, 2)
plt.title("prediction")
for i in range(ntest):
    plt.plot(grid_tx_1, pred_1[i] / factor)
    plt.plot(grid_tx_2, pred_2[i] / factor)
    plt.plot(grid_tx_3, pred_3[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_p$:predictions")
plt.grid()

plt.subplot(2, 2, 3)
plt.title("residuals")
for i in range(ntest):
    plt.plot(grid_tx_1, residual_1[i] / factor)
    plt.plot(grid_tx_2, residual_2[i] / factor)
    plt.plot(grid_tx_3, residual_3[i] / factor)
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
plt.savefig('ex4/vanilla_loss_300.png')
