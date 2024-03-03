
import torch
import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt


# approximate exact solution of 1d singularly perturbed advection diffusion by finite difference method on Shishkinmesh.
def FD_multi_distribution(f, eps, meshtype='Shishkin'):
    def p(x):
        return 0 # p(x)>=alpha=1

    def q(x):
        return np.exp(x)

    alpha = 1
    N = f.shape[-1]
    Nd = f.shape[0]
    grid = np.linspace(0, 1, N)
    yS = np.zeros((Nd, N))
    y = np.zeros((Nd, N))

    for k in range(Nd):
        epsilon= eps[k]
        sigma = min(1 / 2, 2 * epsilon * np.log(N) / alpha)
        gridS = np.hstack(
            (np.linspace(0, 1 - sigma, int((N - 1) / 2) + 1), np.linspace(1 - sigma, 1, int((N - 1) / 2) + 1)[1:]))
        fS = interpolate.interp1d(np.linspace(0, 1, N), f)(gridS)
        h1 = (1 - sigma) / ((N - 1) / 2)
        h2 = sigma / ((N - 1) / 2)
        U = np.zeros((N - 2, N - 2))
        U[0, :2] = np.array([2 * epsilon / (h1 ** 2) + p(h1) / h1 + q(h1), -epsilon / (h1 ** 2)])
        U[-1, -2:] = np.array(
            [-epsilon / (h2 ** 2) - p(1 - h2) / h2, 2 * epsilon / (h2 ** 2) + p(1 - h2) / h2 + q(1 - h2)])

        Nm = int((N - 3) / 2)  # here N must be odd
        for i in range(1, Nm):
            x_i = (i + 1) * h1
            p1 = -epsilon / (h1 ** 2) - p(x_i) / h1
            r1 = 2 * epsilon / (h1 ** 2) + p(x_i) / h1 + q(x_i)
            q1 = -epsilon / (h1 ** 2)
            U[i, i - 1:i + 2] = np.array([p1, r1, q1])
        x_i = (Nm + 1) * h1
        U[Nm, Nm - 1:Nm + 2] = np.array(
            [-2 * epsilon / (h1 * (h1 + h2)) - p(x_i) / h1, 2 * epsilon / (h1 * h2) + p(x_i) / h1 + q(x_i),
             -2 * epsilon / (h2 * (h1 + h2))])
        for i in range(Nm + 1, N - 3):
            x_i = (Nm + 1) * h1 + (i - Nm) * h2
            p2 = -epsilon / (h2 ** 2) - p(x_i) / h2
            r2 = 2 * epsilon / (h2 ** 2) + p(x_i) / h2 + q(x_i)
            q2 = -epsilon / (h2 ** 2)
            U[i, i - 1:i + 2] = np.array([p2, r2, q2])

        B = np.zeros(N - 2)
        B[:] = fS[k, 1:-1]
        B = B.T

        yS[k, 0] = 0
        yS[k, 1:-1] = np.linalg.solve(U, B).flatten()
        yS[k, -1] = 0
        y[k,:] = interpolate.interp1d(gridS, yS[k,:])(grid)
    if meshtype == 'Shishkin':
        return yS
    else:
        return y


# generate random functions(1d) default dim=1001
def generate(samples=1000, begin=0, end=1, random_dim=101, out_dim=1001, length_scale=1, interp="cubic", A=0):
    space = GRF(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples, A)
    x_grid = np.linspace(begin, end, out_dim)
    x_data = space.eval_u(features, x_grid[:, None])
    return x_data  # X_data.shape=(samples,out_dim)，每一行表示一个GRF在x_grid上的取值，共有samples个GRF


# Gaussian Random Field
class GRF(object):
    def __init__(self, begin=0, end=1, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(begin, end, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n, A):
        """Generate `n` random feature vectors.
        """
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T + A

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])

        res = np.zeros((ys.shape[0], sensors.shape[0]))
        for i in range(ys.shape[0]):
            res[i, :] = interpolate.interp1d(np.ravel(self.x), ys[i], kind=self.interp, copy=False, assume_sorted=True)(
                sensors).T
        return res

def weighted_mse_loss(y_pred, y_true, weights):
    # 自定义加权MSE损失函数
    squared_errors = torch.square(y_pred - y_true)
    weighted_errors = squared_errors * weights
    return torch.sum(weighted_errors)


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        if y.size()[-1] == 1:
            eps = 0.00001
        else:
            eps = 0
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + eps))
            else:
                return torch.sum(diff_norms / (y_norms + eps))

        return diff_norms / (y_norms + eps)

    def __call__(self, x, y):
        return self.rel(x, y)


# x_data = generate(samples=10)
# grid=np.linspace(0,1,x_data.shape[-1])
# eps = np.zeros((x_data.shape[0],))
# for i in range(x_data.shape[0]):
#     # eps[i] = (i%10+1)*0.001
#     eps[i] = 0.001
# y_data = FD_multi_distribution(x_data,eps)
# alpha=1
# N=x_data.shape[-1]
# for i in range(10):
#     epsilon = eps[i]
#     sigma = min(1 / 2, 2 * epsilon * np.log(N) / alpha)
#     gridS = np.hstack(
#         (np.linspace(0, 1 - sigma, int((N - 1) / 2) + 1), np.linspace(1 - sigma, 1, int((N - 1) / 2) + 1)[1:]))
#     plt.title("$-\epsilon u''+xu=f$")
#     plt.plot(gridS,y_data[i])
#     plt.xlabel("x")
#     plt.ylabel("u")
# plt.grid()
# plt.show()