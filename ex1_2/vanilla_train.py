
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
# from ex1_tfpm import tfpm
# import generate_data_1d as generate_data_1d
from math import sqrt

# importlib.reload(generate_data_1d)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



eps = 0.0001
NS = 641
ntrain = 1000
ntest = 100
factor = 10
learning_rate = 0.0002
epochs = 800
step_size = 60
gamma = 0.6
alpha = 1

# f1 = generate_data_1d.generate(begin=0.0, end=0.5, samples=1100, out_dim=641)
# f2 = generate_data_1d.generate(begin=0.5, end=1.0, samples=1100, out_dim=641)
# f = np.hstack((f1[:, :-1], f2))
# np.save('f.npy', f)
f = np.load('ex1_2/f.npy')
# 对输入的f不需要除以eps，他们只是相差一个共同的常数eps，如果是multi-eps的情形，输入的应该是f/eps
# f = f/eps
N_max = f.shape[-1]

# u = tfpm(f, eps)
# np.save('code/ex1/ex1_u.npy', u)
u = np.load('ex1_2/u.npy')
# for i in range(10):
#     plt.plot(f[i])
# plt.savefig('/home/v-tingdu/code/icann/ex1_exam_f')
# plt.figure()
# for i in range(10):
#     plt.plot(u[i])
# plt.savefig('/home/v-tingdu/code/icann/ex1_exam_u')
u *= factor
f[:, 640:] /= 0.0001
f_train = f[:ntrain, :]
u_train = u[:ntrain, :]


loss_history = dict()
mse_history = []
print("N value : ", NS - 1)


N = ntrain * NS
grid_h = np.linspace(0, 1, N_max)
gridx = np.linspace(0, 1, NS)
f_train = interpolate.interp1d(grid_h, f_train)(gridx)
u_train = interpolate.interp1d(grid_h, u_train)(gridx)
input_loc = np.tile(gridx, ntrain).reshape((N, 1))
input_f = np.repeat(f_train, NS, axis=0)
output = u_train.reshape((N, 1))
max_f = input_f.max(axis=0)
min_f = input_f.min(axis=0)
input_f = (input_f - min_f)/(max_f - min_f)
input_f = torch.Tensor(input_f).to(device)
input_loc = torch.Tensor(input_loc).to(device)
output = torch.Tensor(output).to(device)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc, output),
                                           batch_size=256, shuffle=True)

model = DeepONet(NS,  1).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    for x, l, y in train_loader:
        optimizer.zero_grad()
        out = model(x, l)
        mse = F.mse_loss(out.view(out.numel(), -1), y.view(y.numel(), -1), reduction='mean')
        mse.backward()
        optimizer.step()
        train_mse += mse.item()
    scheduler.step()
    # train_mse /= ntrain
    train_mse /= len(train_loader)
    t2 = default_timer()
    mse_history.append(train_mse)
    print('Epoch {:d}/{:d}, MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1))
    # print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1), end='',
    #       flush=False)

print('Total training time:', default_timer() - start, 's')
loss_history["{}".format(NS)] = mse_history
torch.save(model.state_dict(), 'ex1_2/vanilla_model_640.pt')


dim = 1001  # test resolution, dim must be odd
batch_size = dim
N = ntest * dim
f_test = f[-ntest:, :]
grid_t = np.linspace(0, 1, dim)
u_test = u[-ntest:, :]
f_test = interpolate.interp1d(grid_h, f_test)(gridx)
u_test = interpolate.interp1d(grid_h, u_test)(grid_t)
input_f = np.repeat(f_test, dim, axis=0)
input_loc = np.tile(grid_t, ntest).reshape((N, 1))
output = u_test.reshape((N, 1))
input_f = (input_f - min_f)/(max_f - min_f)
input_f = torch.Tensor(input_f).to(device)
input_loc = torch.Tensor(input_loc).to(device)
output = torch.Tensor(output).to(device)
test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc, output),
                                            batch_size=batch_size, shuffle=False)

pred = torch.zeros((ntest, dim))
index = 0
test_mse = 0
with torch.no_grad():
    for x, l, y in test_h_loader:
        out = model(x, l)
        pred[index] = out.view(-1)
        mse = F.mse_loss(out.view(1, -1), y.view(1, -1), reduction='mean')
        test_mse += mse.item()
        index += 1
    test_mse /= len(test_h_loader)
    print('test error on high resolution: MSE = ', test_mse)


pred = pred.cpu()
residual = pred - u_test
fig = plt.figure(figsize=(8, 6), dpi=150)

plt.subplot(2, 2, 1)
plt.title("ground truth")
for i in range(ntest):
    plt.plot(grid_t, u_test[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_g$:ground truth")
plt.grid()

plt.subplot(2, 2, 2)
plt.title("prediction")
for i in range(ntest):
    plt.plot(grid_t, pred[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_p$:predictions")
plt.grid()

plt.subplot(2, 2, 3)
plt.title("residuals")
for i in range(ntest):
    plt.plot(grid_t, residual[i] / factor)
plt.xlabel("x")
plt.ylabel("$u_p$-$u_g$:residual")
plt.grid()

plt.subplot(2, 2, 4)
plt.title("training loss")
plt.plot(loss_history["{}".format(NS)])
plt.yscale("log")
plt.xlabel("epoch")
plt.grid()

plt.tight_layout()
plt.show(block=True)
plt.savefig('ex1_2/vanilla_loss_640.png')
