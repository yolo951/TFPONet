
import sys 
sys.path.append('D:\pycharm\pycharm_project\TFPONet')

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
import generate_data_1d as generate_data_1d
from math import sqrt

# importlib.reload(generate_data_1d)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



eps = 0.0001
NS = 257
ntrain = 1000
ntest = 100
factor = 10
learning_rate = 0.0002
epochs = 1000
step_size = 60
gamma = 0.6
alpha = 1

# f1 = generate_data_1d.generate(begin=0.0, end=0.5, samples=1100, out_dim=641)
# f2 = generate_data_1d.generate(begin=0.5, end=1.0, samples=1100, out_dim=641)
# f = np.hstack((f1[:, :-1], f2))
# np.save('f.npy', f)
f = np.load('ex1/f.npy')
N_max = f.shape[-1]
u = np.load('ex1/ex1_u.npy')
model = DeepONet(NS,  1).to(device)
model.load_state_dict(torch.load('ex1/vanila_ex1_model_256.pt')) 

u *= factor
f_train = f[:ntrain, :]
u_train = u[:ntrain, :]


loss_history = dict()
mse_history = []
print("N value : ", NS - 1)
N = ntrain * NS
grid_h = np.linspace(0, 1, N_max)
gridx = np.linspace(0, 1, NS)





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


# for i in range(ntest):
#     fig = plt.figure(figsize=(8, 4), dpi=150)
#     plt.subplot(1, 2, 1)
#     plt.title("ground truth")
#     plt.plot(grid_t, u_test[i] / factor, label='ground truth')
#     plt.plot(grid_t, pred[i] / factor, label='prediction')
#     plt.xlabel("x")
#     plt.ylabel("$u_g$:ground truth")
#     plt.legend()
#     plt.grid()

#     plt.subplot(1, 2, 2)
#     plt.title("prediction")
#     plt.plot(grid_t, residual[i] / factor)
#     plt.xlabel("x")
#     plt.ylabel("$u_p$:predictions")
#     plt.grid()

#     plt.tight_layout()
#     plt.show(block=True)
#     plt.savefig('code/ex1/alltestfig/ex1_fig{}.png'.format(i))
