import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import iv
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from math import sqrt

from scipy.stats import multivariate_normal
from scipy import integrate
from scipy import linalg
from scipy import interpolate
from sklearn import gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D


# Gaussian Progress Regression
class GP_regression:
    def __init__(self, length_scale, sigma_f, num_x_samples, observations):
        self.l = length_scale
        self.sigma = sigma_f
        self.observations = {"x": list(), "y": list()}
        self.update_observation(observations)
        self.num_x_samples = num_x_samples
        self.x_samples = np.linspace(self.observations["x"][0], self.observations["x"][1], self.num_x_samples).reshape(-1, 1)

        self.mu = np.zeros_like(self.x_samples)
        self.cov = self.kernel(self.x_samples, self.x_samples)

    def update(self):
        x = np.array(self.observations["x"]).reshape(-1, 1)
        y = np.array(self.observations["y"]).reshape(-1, 1)

        K11 = self.cov
        K22 = self.kernel(x, x)
        K12 = self.kernel(self.x_samples, x)
        K21 = self.kernel(x, self.x_samples)
        K22_inv = np.linalg.inv(K22 + 1e-8 * np.eye(len(x)))

        self.mu = K12.dot(K22_inv).dot(y)
        self.cov = self.kernel(self.x_samples, self.x_samples) - K12.dot(K22_inv).dot(K21)

    def visualize(self, num_gp_samples=3):
        gp_samples = np.random.multivariate_normal(
            mean=self.mu.ravel(),
            cov=self.cov,
            size=num_gp_samples)
        x_sample = self.x_samples.ravel()
        mu = self.mu.ravel()
        # uncertainty = 1.96 * np.sqrt(np.diag(self.cov))

        # plt.figure()
        # # plt.fill_between(x_sample, mu + uncertainty, mu - uncertainty, alpha=0.1)
        # plt.plot(x_sample, mu, label='Mean')
        # for i, gp_sample in enumerate(gp_samples):
        #     plt.plot(x_sample, gp_sample, lw=1, ls='-', label=f'Sample {i + 1}')

        # plt.plot(self.observations["x"], self.observations["y"], 'rx')
        # # plt.legend()
        # plt.grid()
        return gp_samples

    def update_observation(self, observations):
        for x, y in zip(observations["x"], observations["y"]):
            if x not in self.observations["x"]:
                self.observations["x"].append(x)
                self.observations["y"].append(y)


    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.sigma ** 2 * np.exp(-0.5 / self.l ** 2 * dist_matrix)


def dim2_tfpm(f1, f2):
    Nd = f1.shape[0]
    N1 = f1.shape[-1]*2 - 1
    N2 = N1
    # N0 = 20
    ay, by = -1, 1
    ax, bx = -1, 1
    a0x, ap = 0, 1
    bt, mu1, mu2, f0 = 0, 1000, 1, 0
    h = (bx - ax) / (N2-1)
    # N1, N2 = N0 + 1, N0 + 1
    N = N1 * N2
    u1 = np.zeros((Nd, N1, f1.shape[-1]), dtype=np.float64)
    u2 = np.zeros((Nd, N1, f2.shape[-1]), dtype=np.float64)
    J1 = int((N2-1) / 2)
    A = lil_matrix((N, N))
    F = np.zeros(N)
    x = np.arange(ax, bx + h, h)
    y = np.arange(ay, by + h, h)
    x1 = np.arange(ax, a0x + h, h)
    x2 = np.arange(a0x, bx + h, h)
    left = np.arange(0, J1 + 1)
    right = np.arange(J1, N2)
    # mu = mu1 * np.ones((N2, N1))
    # mu[right, :] = mu2
    mu = np.ones((N2, N1))
    mu1 = np.tile(sqrt(1000)*(1-np.square(np.linspace(-1, 0, J1+1))), N2).reshape((N2, -1))
    mu[left, :] = mu1.T
    # mu[left, :] = 100
    mu[right, :] = 1
    sq2 = np.sqrt(2)


    def bessel_matrix(nu, z):
        return np.array([[iv(nu, z_ij) for z_ij in z_i] for z_i in z])


    I_0 = bessel_matrix(0, mu * h)
    I_1 = bessel_matrix(1, mu * h)
    I_2 = bessel_matrix(2, mu * h)
    I_3 = bessel_matrix(3, mu * h)
    I_4 = bessel_matrix(4, mu * h)
    I_0_2 = bessel_matrix(0, sq2 * mu * h)
    I_1_2 = bessel_matrix(1, sq2 * mu * h)
    I_2_2 = bessel_matrix(2, sq2 * mu * h)
    I_3_2 = bessel_matrix(3, sq2 * mu * h)
    I_4_2 = bessel_matrix(4, sq2 * mu * h)

    bt = bt * h

    for j in range(N):
        A[j, j] = 1

    for j in range(1, N2 - 1):
        j1 = j * N1
        for k in range(1, N1 - 1):
            jk = j1 + k
            ind = [jk, jk + N1, jk + 1, jk - N1, jk - 1, jk + N1 + 1, jk - N1 + 1, jk - N1 - 1, jk + N1 - 1]
            s = I_0[j, k] * I_4_2[j, k] + I_0_2[j, k] * I_4[j, k]
            elem = [4 * s] + [-I_4_2[j, k]] * 4 + [-I_4[j, k]] * 4
            A[jk, ind] = elem

            if j == J1 - 1:
                F[jk] += ap * (elem[1] + elem[5] + elem[8])

            if j == J1:
                m1 = mu[j - 1, k] * h
                m2 = mu[j + 1, k] * h
                s1 = I_0[j - 1, k] * I_4_2[j - 1, k] + I_0_2[j - 1, k] * I_4[j - 1, k]
                s2 = I_0[j + 1, k] * I_4_2[j + 1, k] + I_0_2[j + 1, k] * I_4[j + 1, k]
                t1 = I_1[j - 1, k] * I_3_2[j - 1, k] + I_1_2[j - 1, k] * I_3[j - 1, k]
                t2 = I_1[j + 1, k] * I_3_2[j + 1, k] + I_1_2[j + 1, k] * I_3[j + 1, k]
                C01 = m1 * I_1[j - 1, k] * I_4_2[j - 1, k] / s1 / 4
                C02 = m1 * I_1[j - 1, k] * I_4[j - 1, k] / s1 / 4
                C11 = (m1 * I_0[j - 1, k] - I_1[j - 1, k]) * I_3_2[j - 1, k] / t1 / 2
                C12 = (m1 * I_0[j - 1, k] - I_1[j - 1, k]) * I_3[j - 1, k] / t1 / 2 / sq2
                C2 = (m1 * I_1[j - 1, k] - 2 * I_2[j - 1, k]) / I_2[j - 1, k] / 4
                C31 = (m1 * I_2[j - 1, k] - 3 * I_3[j - 1, k]) * I_1_2[j - 1, k] / t1 / 2
                C32 = (m1 * I_2[j - 1, k] - 3 * I_3[j - 1, k]) * I_1[j - 1, k] / t1 / 2 / sq2
                C41 = (m1 * I_3[j - 1, k] - 4 * I_4[j - 1, k]) * I_0_2[j - 1, k] / s1 / 4
                C42 = (m1 * I_3[j - 1, k] - 4 * I_4[j - 1, k]) * I_0[j - 1, k] / s1 / 4
                D01 = m1 * I_1[j + 1, k] * I_4_2[j + 1, k] / s1 / 4
                D02 = m1 * I_1[j + 1, k] * I_4[j + 1, k] / s1 / 4
                D11 = (m1 * I_0[j + 1, k] - I_1[j + 1, k]) * I_3_2[j + 1, k] / t1 / 2
                D12 = (m1 * I_0[j + 1, k] - I_1[j + 1, k]) * I_3[j + 1, k] / t1 / 2 / sq2
                D2 = (m1 * I_1[j + 1, k] - 2 * I_2[j + 1, k]) / I_2[j + 1, k] / 4
                D31 = (m1 * I_2[j + 1, k] - 3 * I_3[j + 1, k]) * I_1_2[j + 1, k] / t1 / 2
                D32 = (m1 * I_2[j + 1, k] - 3 * I_3[j + 1, k]) * I_1[j + 1, k] / t1 / 2 / sq2
                D41 = (m1 * I_3[j + 1, k] - 4 * I_4[j + 1, k]) * I_0_2[j + 1, k] / s1 / 4
                D42 = (m1 * I_3[j + 1, k] - 4 * I_4[j + 1, k]) * I_0[j + 1, k] / s1 / 4
                F[jk] = bt + ap * (C01 + 2 * C02 + C11 + 2 * C12 + C2 + C31 - 2 * C32 + C41 - 2 * C42)
                idx = np.array([jk + N1, jk + 1, jk - N1, jk - 1, jk + N1 + 1, jk - N1 + 1, jk - N1 - 1, jk + N1 - 1])
                ind = np.concatenate((idx - N1, idx + N1))
                elem1 = [C01 + C11 + C2 + C31 + C41, C01 - C2 + C41, C01 - C11 + C2 - C31 + C41, C01 - C2 + C41,
                        C02 + C12 - C32 - C42, C01 - C12 + C32 - C42, C01 - C11 + C31 - C41, C01 + C12 - C32 - C42]
                elem2 = [D01 - D11 + D2 - D31 + D41, D01 - D2 + D41, D01 + D11 + D2 + D31 + D41, D01 - D2 + D41,
                        D02 - D12 + D32 - D42, D01 + D12 - D32 - D42, D01 + D11 - D31 - D41, D01 - D12 + D32 - D42]
                A[jk, ind[:8]] = -np.array(elem1)
                A[jk, ind[8:]] = np.array(elem2)

    # Boundary conditions
    for k in range(Nd):
        F[0:J1*N1:N1] = f1[k, :J1]
        F[N1-1:J1*N1:N1] = f1[k, :J1]
        F[((J1+1)*N1-1):N:N1] = f2[k, :]
        F[J1 * N1:N - N1 + 1:N1] = f2[k, :]

        u = spsolve(A.tocsr(), F)
        u = np.reshape(u, (N1, N2)).T
        u2[k] = u[:, right]
        u1[k, :, :J1] = u[:, :J1]
        u1[k, :, J1] = u[:, J1] - ap
    return u1, u2

def data_generate(samples, out_dim, length_scale=0.2, sigma_f=0.5):
    num_x_samples = int(out_dim/2)+1
    gp1 = GP_regression(length_scale, sigma_f, num_x_samples=num_x_samples, observations={"x": [-1, 0], "y": [0, 0]})
    gp1.update()
    f1 = gp1.visualize(samples)
    gp1 = GP_regression(length_scale, sigma_f, num_x_samples=num_x_samples, observations={"x": [0, 1], "y": [1, 0]})
    gp1.update()
    f2 = gp1.visualize(samples)
    ## f1 = np.zeros((1, 11))
    ## f2 = np.cos(np.pi/2*np.linspace(0, 1, 11)).reshape((1,-1))
    # u1, u2 = dim2_tfpm(f1, f2)
    # [x, y] = np.meshgrid(np.linspace(-1, 0, num_x_samples), np.linspace(-1, 1, out_dim))
    # fig = plt.figure()
    # axs = fig.add_subplot(111, projection='3d')
    # axs.plot_surface(x, y, u1[0], cmap='gray')
    # axs.view_init(elev=-10., azim=30)
    # [x, y] = np.meshgrid(np.linspace(0, 1, num_x_samples), np.linspace(-1, 1, out_dim))
    # axs.plot_surface(x, y, u2[0], cmap='gray')
    # plt.show()
    return f1, f2

# data_generate(1, 101)













# hmin = 1/128
# [xi, yi] = np.meshgrid(np.arange(a0x, bx + hmin, hmin), np.arange(ay, by + hmin, hmin))
# interp_func = interp2d(x2, y, u[:, right], kind='cubic')
# zi = interp_func(xi[0, :], yi[:, 0])

# fig = plt.figure()
# axs = fig.add_subplot(111, projection='3d')
# axs.plot_surface(xi, yi, zi, cmap='gray')
# axs.view_init(elev=-10., azim=30)

# u[:, J1] = u[:, J1] - ap
# [xi, yi] = np.meshgrid(np.arange(ax, a0x + hmin, hmin), np.arange(ay, by + hmin, hmin))
# interp_func = interp2d(x1, y, u[:, left], kind='cubic')
# zi = interp_func(xi[0, :], yi[:, 0])

# axs.plot_surface(xi, yi, zi, cmap='gray')
# plt.show()
