import sys

sys.path.append('/home/v-tingdu/code/')
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from timeit import default_timer
from Adam import Adam
import tfponet

importlib.reload(tfponet)
from tfponet import TFPONet2D
from scipy import interpolate
from scipy.special import iv
from math import sqrt

import ex2_generate_data


# importlib.reload(generate_data_1d)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# q(x,y)=mu(x,y)**2
# -Delta u + q(x,y)*u = 0
def mu(x, y):
    return np.where(x <= 0, sqrt(1000)* (1 - x ** 2), 1)


def get_modify(x, y):
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    h = y[1] - y[0]
    ab = np.ones((len(points), 2))  # 目前只输入I0,I1
    for i in range(len(points)):
        ab[i, 0] = iv(0, h * mu(points[i][0], points[i][1])/sqrt(2))
        ab[i, 1] = iv(1, h * mu(points[i][0], points[i][1])/sqrt(2))
    return points, ab


ntrain = 500
ntest = 50
learning_rate = 0.00004
epochs = 600
step_size = 100
gamma = 0.4
ap = 1
factor = 10

f1 = np.load('ex2/ex2_f1.npy')
f2 = np.load('ex2/ex2_f2.npy')
## f1.shape[-1]=f2.shape[-1]=2**6+1
# f1, f2 = ex2_generate_data.data_generate(ntrain+ntest, 2**7+1)
# np.save('/home/v-tingdu/code/icann/ex2_f1.npy',f1)
# np.save('/home/v-tingdu/code/icann/ex2_f2.npy',f2)
# u1, u2 = ex2_generate_data.dim2_tfpm(f1, f2)
# np.save('/home/v-tingdu/code/ex2/ex2_u1.npy', u1)
# np.save('/home/v-tingdu/code/ex2/ex2_u2.npy', u2)
u1 = np.load('ex2/ex2_u1.npy')
u2 = np.load('ex2/ex2_u2.npy')

u1 *= factor
u2 *= factor

loss_history = dict()

NS = 64
N2_max = f1.shape[-1]
mse_history = []
mse_jump_history = []
mse_jump_deriv_history = []

f1_train = interpolate.interp1d(np.linspace(-1, 0, N2_max), f1[:ntrain])(np.linspace(-1, 0, NS + 1))
f2_train = interpolate.interp1d(np.linspace(0, 1, N2_max), f2[:ntrain])(np.linspace(0, 1, NS + 1))
f_train = np.concatenate((f1_train, f2_train[:, 1:]), axis=1)
N = ntrain * (NS + 1) ** 2

x_h = np.linspace(-1, 1, 2 * N2_max - 1)
y_h = np.linspace(-1, 1, 2 * N2_max - 1)
x1 = np.linspace(-1, 0, NS + 1)
x2 = np.linspace(0, 1, NS + 1)
y = np.linspace(-1, 1, NS + 1)
u_train_1 = np.zeros((ntrain, NS + 1, NS + 1))  # shape=(ntrain, dimX, dimY)
u_train_2 = np.zeros((ntrain, NS + 1, NS + 1))
for i in range(ntrain):
    # z.size = (x.size, y.size)
    u_train_1[i] = interpolate.RectBivariateSpline(x_h[:N2_max], y_h, u1[i].T)(x1, y).T
    u_train_2[i] = interpolate.RectBivariateSpline(x_h[N2_max - 1:], y_h, u2[i].T)(x2, y).T
input_loc_1, ab_1 = get_modify(x1, y)
input_loc_2, ab_2 = get_modify(x2, y)
input_f = np.repeat(f_train, (NS + 1) ** 2, axis=0)
input_loc_1 = np.tile(input_loc_1, (ntrain, 1))
input_loc_2 = np.tile(input_loc_2, (ntrain, 1))
input_modify_1 = np.tile(ab_1, (ntrain, 1))
input_modify_2 = np.tile(ab_2, (ntrain, 1))
output_1 = u_train_1.reshape((N, 1))
output_2 = u_train_2.reshape((N, 1))
# max_modify_1 = ab_1.max(axis=0)
# min_modify_1 = ab_1.min(axis=0)
# max_modify_2 = ab_2.max(axis=0)
# min_modify_2 = ab_2.min(axis=0)
# input_modify_1 = (input_modify_1 - min_modify_1)/(max_modify_1 - min_modify_1)
# input_modify_2 = (input_modify_2 - min_modify_2)/(max_modify_2 - min_modify_2)
input_f = torch.Tensor(input_f).to(device)
input_loc_1 = torch.Tensor(input_loc_1).to(device)
input_loc_2 = torch.Tensor(input_loc_2).to(device)
input_modify_1 = torch.Tensor(input_modify_1).to(device)
input_modify_2 = torch.Tensor(input_modify_2).to(device)
output_1 = torch.Tensor(output_1).to(device)
output_2 = torch.Tensor(output_2).to(device)
disc_point, disc_point_modify = get_modify(np.array([0.5]), y)
disc_point = torch.tensor(disc_point, dtype=torch.float32).to(device)
disc_point_modify = torch.tensor(disc_point_modify, dtype=torch.float32).to(device)
# disc_point_modify1 = (disc_point_modify - min_modify_1)/(max_modify_1 - min_modify_1)
# disc_point_modify2 = (disc_point_modify - min_modify_2)/(max_modify_2 - min_modify_2)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc_1, input_loc_2,
                                                                          input_modify_1, input_modify_2, output_1,
                                                                          output_2),
                                           batch_size=520, shuffle=True)
# batch_size = ratio*(NS+1)

model_1 = TFPONet2D(2*NS + 1, 2).to(device)
model_2 = TFPONet2D(2*NS + 1, 2).to(device)
model_1.load_state_dict(torch.load('ex2/ex2_model1.pt'))
model_2.load_state_dict(torch.load('ex2/ex2_model2.pt'))
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
    for x, l_1, l_2, e_1, e_2, y_1, y_2 in train_loader:
        # handle = nvmlDeviceGetHandleByIndex(0)
        # util_start = nvmlDeviceGetUtilizationRates(handle)
        ratio = int(x.shape[0] / disc_point.shape[0])
        point = torch.tile(disc_point, (ratio, 1))
        point_modify = torch.tile(disc_point_modify, (ratio, 1))

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        out_1 = model_1(x, l_1, e_1[:, 0].reshape((-1, 1)), e_1[:, 1].reshape((-1, 1)))
        out_2 = model_2(x, l_2, e_2[:, 0].reshape((-1, 1)), e_2[:, 1].reshape((-1, 1)))
        mse_1 = F.mse_loss(out_1.view(out_1.numel(), -1), y_1.view(y_1.numel(), -1), reduction='mean')
        mse_2 = F.mse_loss(out_2.view(out_2.numel(), -1), y_2.view(y_2.numel(), -1), reduction='mean')
        mse = mse_1 + mse_2
        out_1 = model_1(x, point, point_modify[:, 0].reshape((-1, 1)), point_modify[:, 1].reshape((-1, 1)))
        out_2 = model_2(x, point, point_modify[:, 0].reshape((-1, 1)), point_modify[:, 1].reshape((-1, 1)))
        mse_jump = F.mse_loss(out_2 - ap * factor, out_1, reduction='mean')
        mse += 0.2 * mse_jump
        mse.backward()
        optimizer_1.step()
        optimizer_2.step()
        train_mse += mse.item()
        train_mse_jump += mse_jump.item()
        
    scheduler_1.step()
    scheduler_2.step()
    # train_mse /= ntrain
    train_mse /= len(train_loader)
    t2 = default_timer()
    mse_history.append(train_mse)
    mse_jump_history.append(train_mse_jump / len(train_loader))
    mse_jump_deriv_history.append(train_mse_jump_deriv / len(train_loader))
    print('Epoch {:d}/{:d}, MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1))
    if not (ep+1)%100:
        torch.save(model_1.state_dict(), 'ex2/ex2_model1_update.pt')
        torch.save(model_2.state_dict(), 'ex2/ex2_model2_update.pt')
        torch.save(torch.tensor(mse_history), 'ex2/mse_history.pt')
        torch.save(torch.tensor(mse_jump_history), 'ex2/mse_jump_history.pt')
        print(ep)
    # print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1), end='',
    #       flush=False)

print('Total training time:', default_timer() - start, 's')
loss_history["{}".format(NS)] = mse_history

torch.save(model_1.state_dict(), 'ex2/ex2_model1_update.pt')
torch.save(model_2.state_dict(), 'ex2/ex2_model2_update.pt')

dim = 128  # test resolution, dim must be odd
N = ntest * (dim + 1) ** 2
batch_size = (dim + 1) ** 2
f1_test = interpolate.interp1d(np.linspace(-1, 0, N2_max), f1[ntrain:ntrain + ntest])(np.linspace(-1, 0, NS + 1))
f2_test = interpolate.interp1d(np.linspace(0, 1, N2_max), f2[ntrain:ntrain + ntest])(np.linspace(0, 1, NS + 1))
f_test = np.concatenate((f1_test, f2_test[:, 1:]), axis=1)
x1 = np.linspace(-1, 0, dim + 1)
x2 = np.linspace(0, 1, dim + 1)
y = np.linspace(-1, 1, dim + 1)
u_test_1 = np.zeros((ntest, dim + 1, dim + 1))
u_test_2 = np.zeros((ntest, dim + 1, dim + 1))
for i in range(ntest):
    u_test_1[i] = interpolate.RectBivariateSpline(x_h[:N2_max], y_h, u1[ntrain + i].T)(x1, y).T
    u_test_2[i] = interpolate.RectBivariateSpline(x_h[N2_max - 1:], y_h, u2[ntrain + i].T)(x2, y).T
input_loc_1, ab_1 = get_modify(x1, y)
input_loc_2, ab_2 = get_modify(x2, y)
input_f = np.repeat(f_test, (dim + 1) ** 2, axis=0)
input_loc_1 = np.tile(input_loc_1, (ntest, 1))
input_loc_2 = np.tile(input_loc_2, (ntest, 1))
input_modify_1 = np.tile(ab_1, (ntest, 1))
input_modify_2 = np.tile(ab_2, (ntest, 1))
output_1 = u_test_1.reshape((N, 1))
output_2 = u_test_2.reshape((N, 1))
# input_modify_1 = (input_modify_1 - min_modify_1)/(max_modify_1 - min_modify_1)
# input_modify_2 = (input_modify_2 - min_modify_2)/(max_modify_2 - min_modify_2)
input_f = torch.Tensor(input_f).to(device)
input_loc_1 = torch.Tensor(input_loc_1).to(device)
input_loc_2 = torch.Tensor(input_loc_2).to(device)
input_modify_1 = torch.Tensor(input_modify_1).to(device)
input_modify_2 = torch.Tensor(input_modify_2).to(device)
output_1 = torch.Tensor(output_1).to(device)
output_2 = torch.Tensor(output_2).to(device)
test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc_1, input_loc_2,
                                                                           input_modify_1, input_modify_2, output_1,
                                                                           output_2),
                                            batch_size=batch_size, shuffle=False)

pred_1 = torch.zeros((ntest, (dim + 1) ** 2))
pred_2 = torch.zeros((ntest, (dim + 1) ** 2))
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

plt.title("training loss")
plt.plot(loss_history["{}".format(NS)], label='total loss')
plt.plot(mse_jump_history, label='jump loss')
plt.yscale("log")
plt.xlabel("epoch")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show(block=True)
plt.savefig('ex2/ex2_loss.png')
