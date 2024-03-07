
import sys
sys.path.append('D:\pycharm\pycharm_project\TFPONet')
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tfponet import TFPONet2D
from scipy import interpolate
from scipy.special import iv
import pdb

def mu(x, y):
    return np.where(x <= 0, 100*(1-x**2)+1, 0.1)


def get_modify(x, y):
    X,Y=np.meshgrid(x,y)
    points=np.vstack([X.ravel(),Y.ravel()]).T
    h = y[1]-y[0]
    ab = np.ones((len(points), 2)) # 目前只输入I0,I1
    for i in range(len(points)):
        ab[i, 0] = iv(0, h*mu(points[i][0], points[i][1]))
        ab[i, 1] = iv(1, h*mu(points[i][0], points[i][1]))
    return points, ab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntrain = 500
ntest = 50
factor = 10
NS = 64
f1 = np.load('ex2/ex2_f1.npy')
f2 = np.load('ex2/ex2_f2.npy')
u1 = np.load('ex2/ex2_u1.npy')
u2 = np.load('ex2/ex2_u2.npy')
u1 *= factor
u2 *= factor
N2_max = f1.shape[-1]
x_h = np.linspace(-1, 1, 2*N2_max-1)
y_h = np.linspace(-1, 1, 2*N2_max-1)
model_1 = TFPONet2D(2*NS + 1,  2).to(device)
model_1.load_state_dict(torch.load('ex2/ex2_model1_update.pt'))
model_2 =  TFPONet2D(2*NS + 1,  2).to(device)
model_2.load_state_dict(torch.load('ex2/ex2_model2_update.pt'))
dim = 64
N = ntest*(dim+1)**2
batch_size = (dim+1)**2
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
# pdb.set_trace()
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
                                                                          input_modify_1, input_modify_2, output_1, output_2),
                                            batch_size=batch_size, shuffle=False)

pred_1 = torch.zeros((ntest, (dim+1)**2))
pred_2 = torch.zeros((ntest, (dim+1)**2))
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

tfponet_pred_1 = pred_1.reshape((ntest, dim+1, dim+1)).numpy()
tfponet_pred_2 = pred_2.reshape((ntest, dim+1, dim+1)).numpy()
pred_1 /= factor
pred_2 /= factor

from deeponet import DeepONet2D
model_1 = DeepONet2D(2*NS + 1,  2).to(device)
model_1.load_state_dict(torch.load('ex2/vanilla_model1.pt'))
model_2 =  DeepONet2D(2*NS + 1,  2).to(device)
model_2.load_state_dict(torch.load('ex2/vanilla_model2.pt'))
def get_modify(x, y):
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    return points
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
input_loc_1 = get_modify(x1, y)
input_loc_2 = get_modify(x2, y)
input_f = np.repeat(f_test, (dim + 1) ** 2, axis=0)
input_loc_1 = np.tile(input_loc_1, (ntest, 1))
input_loc_2 = np.tile(input_loc_2, (ntest, 1))
output_1 = u_test_1.reshape((N, 1))
output_2 = u_test_2.reshape((N, 1))
input_f = torch.Tensor(input_f).to(device)
input_loc_1 = torch.Tensor(input_loc_1).to(device)
input_loc_2 = torch.Tensor(input_loc_2).to(device)
output_1 = torch.Tensor(output_1).to(device)
output_2 = torch.Tensor(output_2).to(device)
test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc_1, input_loc_2, output_1,
                                                                           output_2),
                                            batch_size=batch_size, shuffle=False)

pred_1 = torch.zeros((ntest, (dim + 1) ** 2))
pred_2 = torch.zeros((ntest, (dim + 1) ** 2))
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

deeponet_pred_1 = pred_1.reshape((ntest, dim+1, dim+1)).numpy()
deeponet_pred_2 = pred_2.reshape((ntest, dim+1, dim+1)).numpy()
u_test_1 /= factor
u_test_2 /= factor
deeponet_pred_1 /= factor
deeponet_pred_2 /= factor

for i in range(34,35):
    tfponet_residual1 = u_test_1[i]-tfponet_pred_1[i]
    tfponet_residual2 = u_test_2[i]-tfponet_pred_2[i]
    deeponet_residual1 = u_test_1[i] - deeponet_pred_1[i]
    deeponet_residual2 = u_test_2[i] - deeponet_pred_2[i]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    [X, Y] = np.meshgrid(x1[:-1], y)
    img1 = axs[0].contourf(X, Y, np.abs(tfponet_residual1[:, :-1]), np.arange(0, 0.021, 0.0001), extend='neither', cmap='cividis')
    [X, Y] = np.meshgrid(x2[1:], y)
    img1 = axs[0].contourf(X, Y, np.abs(tfponet_residual2[:, 1:]), np.arange(0, 0.021, 0.0001), extend='neither', cmap='cividis')
    [X, Y] = np.meshgrid(x1[:-1], y)
    # cbar = fig.colorbar(img1, ax=axs, orientation='vertical', shrink=0.6, aspect=20)
    # cbar.set_ticks(np.arange(0, 0.04, 0.01))
    img2 = axs[1].contourf(X, Y, np.abs(deeponet_residual1[:, :-1]), np.arange(0, 0.021, 0.0001), extend='neither', cmap='cividis')
    [X, Y] = np.meshgrid(x2[1:], y)
    img2 = axs[1].contourf(X, Y, np.abs(deeponet_residual2[:, 1:]), np.arange(0, 0.021, 0.0001), extend='neither', cmap='cividis')
    cbar = fig.colorbar(img2, ax=axs, orientation='vertical', shrink=0.6, aspect=20)
    cbar.set_ticks(np.arange(0, 0.021, 0.005))
    plt.savefig('ex2/comparefig/test{}.png'.format(i))