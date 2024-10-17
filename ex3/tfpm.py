
import sys
sys.path.append('D:\pycharm\pycharm_project\TFPONet')
import numpy as np
from scipy.special import airy
from scipy.integrate import quad
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos, e
from scipy import interpolate
import generate_data_1d

# solve equation -u''(x)+q(x)u(x)=F(x)
def discontinuous_tfpm(f):

    def q(x):
        return np.where(x<=0.5, 5000.0, 100*(4+32*x))

    def q2(x):
        return 100*(4+32*x)

    def F(x):
        return interpolate.interp1d(np.linspace(0, 1, N), f[k])(x)
        # return interpolate_f(x)[k]

    def integrand_linear(s):  # for -u''=f
        G = np.where(y >= s, s-y, 0)
        return F(s)*G

    def integrand_sin(s):  # for -u''+bu = f, b<0
        G = 1/sin(1)*np.where(y >= s, np.sin(1-y)*sin(s), np.sin(1-s)*sin(y))
        return F(s) * G

    def integrandp_sin(s):
        G = 1/sin(1)*np.where(y >= s, -np.cos(1-y)*sin(s), np.sin(1-s)*cos(y))
        return F(s) * G

    def integrand_sinh(s):  # for -u''+bu = f, b>0
        G = 1/2/sqrt(b)*np.where(y >= s, np.sinh(sqrt(b)*(s-y)), np.sinh(sqrt(b)*(y-s)))
        return F(s) * G

    def integrandp_sinh(s):
        G = 1/2*np.where(y >= s, -np.cosh(sqrt(b)*(s-y)), np.cosh(sqrt(b)*(y-s)))
        return F(s) * G

    def integrand_airy(s):  # for -u'' + (ax+b)u = f
        G = np.pi/2*np.where(z >= s, airy(s)[2] * airy(z)[0] - airy(s)[0] * airy(z)[2],
                              airy(s)[0] * airy(z)[2] - airy(s)[2] * airy(z)[0])
        return np.power(np.abs(a), -2/3)*G*F(s/np.cbrt(a)-b/a)

    def integrandp_airy(s):
        G = np.pi/2 * np.where(z >= s, airy(s)[2] * airy(z)[1] - airy(s)[0] * airy(z)[3],
                              airy(s)[0] * airy(z)[3] - airy(s)[2] * airy(z)[1])
        return 1/np.cbrt(a)*G*F(s/np.cbrt(a)-b/a)

    N = f.shape[-1]
    Nd = f.shape[0]
    # interpolate_f = interpolate.interp1d(np.linspace(0, 1, N), f)
    x = np.zeros((Nd, N), dtype=np.float64)
    x1 = np.zeros((Nd, int((N+1)/2)), dtype=np.float64)
    x2 = np.zeros((Nd, int((N + 1) / 2)), dtype=np.float64)
    U = np.zeros((2*(N-1), 2*(N-1)), dtype=np.float64)
    B = np.zeros((2*(N-1),), dtype=np.float64)
    for k in range(Nd):
        grid = np.linspace(0, 1, N)
        cRight = q(grid)
        cLeft = q(grid)
        cRight[int((N-1)/2)] = q2(grid[int((N-1)/2)])
        # interp_coeff = get_coeff(grid)
        c = np.polyfit(grid[:2], np.array([cRight[0], cLeft[1]]), 1)
        a, b = c[0], c[1]
        # a, b = interp_coeff[0]
        if abs(a) < 1e-8:
            if abs(b) < 1e-8:
                lambda1 = 1.0
                mu1 = grid[0]
                y = grid[0]
                F1 = quad(integrand_linear, grid[0], grid[1])[0]
            elif b > 0:
                # b = abs(b)
                lambda1 = np.exp(grid[0] * sqrt(b))
                mu1 = np.exp(-grid[0] * sqrt(b))
                y = grid[0]
                F1 = quad(integrand_sinh, grid[0], grid[1])[0]
            else:
                b = abs(b)
                lambda1 = np.sin(grid[0] * sqrt(b))
                mu1 = np.cos(grid[0] * sqrt(b))
                y = grid[0]
                F1 = quad(integrand_sin, grid[0], grid[1])[0]
        else:
            z1 = q(grid[0]) * np.power(np.abs(a), -2 / 3)  # 如果直接使用np.power(a, -2/3),可能会计算出来复数
            z2 = q(grid[1]) * np.power(np.abs(a), -2 / 3)
            lambda1 = airy(z1)[0]
            mu1 = airy(z1)[2]
            z = z1
            F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
        U[0,:2] = np.array([-lambda1, -mu1])
        B[0] = F1
        coeff = []
        for i in range(1, 2*N-3, 2):
            c= np.polyfit(grid[i//2:i//2+2], np.array([cRight[i//2], cLeft[i//2+1]]), 1)
            a1, b1 = c[0], c[1]
            c = np.polyfit(grid[i//2+1:i//2+3], np.array([cRight[i//2+1], cLeft[i//2+2]]), 1)
            a2, b2 = c[0], c[1]
            # a1, b1 = interp_coeff[i // 2]
            # a2, b2 = interp_coeff[i // 2 + 1]
            if abs(a1) < 1e-8:
                if abs(b1) < 1e-8:
                    lambda1, gamma1, delta1 = 1.0, 0.0, 1.0
                    mu1 = grid[i//2+1]
                    y = grid[i//2 + 1]
                    F1 = quad(integrand_linear, grid[i//2], grid[i//2+1])[0]
                    G1 = -quad(F, grid[i//2], grid[i//2+1])[0]
                elif b1>0:
                    # b1 = abs(b1)
                    # b2 = abs(b2)
                    lambda1 = np.exp(sqrt(b1) * grid[i // 2 + 1])
                    gamma1 = sqrt(b1) * lambda1
                    mu1 = np.exp(-sqrt(b1) * grid[i // 2 + 1])
                    delta1 = -sqrt(b1) * mu1
                    y, b = grid[i // 2 + 1], b1
                    F1 = quad(integrand_sinh, grid[i // 2], grid[i // 2 + 1])[0]
                    G1 = quad(integrandp_sinh, grid[i // 2], grid[i // 2 + 1])[0]
                else:
                    b1 = abs(b1)
                    lambda1 = np.sin(sqrt(b1) * grid[i // 2 + 1])
                    gamma1 = sqrt(b1) * np.cos(sqrt(b1) * grid[i // 2 + 1])
                    mu1 = np.cos(sqrt(b1) * grid[i // 2 + 1])
                    delta1 = -sqrt(b1) * np.sin(sqrt(b1) * grid[i // 2 + 1])
                    y, b = grid[i // 2 + 1], b1
                    F1 = quad(integrand_sin, grid[i // 2], grid[i // 2 + 1])[0]
                    G1 = quad(integrandp_sin, grid[i // 2], grid[i // 2 + 1])[0]
            else:
                z1 = cRight[i//2] * np.power(np.abs(a1), -2 / 3)
                z2 = cLeft[i//2+1]* np.power(np.abs(a1), -2 / 3)
                lambda1, gamma1, mu1, delta1 = airy(z2)
                gamma1 *= np.cbrt(a1)
                delta1 *= np.cbrt(a1)
                a, b, z = a1, b1, z2
                F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
                G1 = quad(integrandp_airy, z1, z2)[0] if z2 >= z1 else quad(integrandp_airy, z2, z1)[0]
            if abs(a2) < 1e-8:
                if abs(b2) < 1e-8:
                    lambda2, gamma2, delta2 = 1, 0, 1
                    mu2 = grid[i//2+1]
                    y = grid[i//2 + 1]
                    F2 = quad(integrand_linear, grid[i//2+1], grid[i//2+2])[0]
                    G2 = 0.0
                elif b2>0:
                    lambda2 = np.exp(sqrt(b2) * grid[i // 2 + 1])
                    gamma2 = sqrt(b2) * lambda2
                    mu2 = np.exp(-sqrt(b2) * grid[i // 2 + 1])
                    delta2 = -sqrt(b2) * mu2
                    y, b = grid[i // 2 + 1], b2
                    F2 = quad(integrand_sinh, grid[i // 2 + 1], grid[i // 2 + 2])[0]
                    G2 = quad(integrandp_sinh, grid[i // 2 + 1], grid[i // 2 + 2])[0]
                else:
                    b2 = abs(b2)
                    lambda2 = np.sin(sqrt(b2) * grid[i // 2 + 1])
                    gamma2 = sqrt(b2) * np.cos(sqrt(b2) * grid[i // 2 + 1])
                    mu2 = np.cos(sqrt(b2) * grid[i // 2 + 1])
                    delta2 = -sqrt(b2) * np.sin(sqrt(b2) * grid[i // 2 + 1])
                    y, b = grid[i // 2 + 1], b2
                    F2 = quad(integrand_sinh, grid[i // 2 + 1], grid[i // 2 + 2])[0]
                    G2 = quad(integrandp_sinh, grid[i // 2 + 1], grid[i // 2 + 2])[0]
            else:
                z3 = cRight[i//2+1] * np.power(np.abs(a2), -2 / 3)
                z4 = cLeft[i//2+2] * np.power(np.abs(a2), -2 / 3)
                lambda2, gamma2, mu2, delta2 = airy(z3)
                gamma2 *= np.cbrt(a2)
                delta2 *= np.cbrt(a2)
                a, b, z = a2, b2, z3
                F2 = quad(integrand_airy, z3, z4)[0] if z4 >= z3 else quad(integrand_airy, z4, z3)[0]
                G2 = quad(integrandp_airy, z3, z4)[0] if z4 >= z3 else quad(integrandp_airy, z4, z3)[0]
            B[i] = F2 - F1
            B[i+1] = G2 - G1
            U[i, i-1:i+3] = np.array([lambda1, mu1, -lambda2, -mu2])
            U[i+1, i - 1:i + 3] = np.array([gamma1, delta1, -gamma2, -delta2])
            coeff.append([lambda1, mu1, F1])
        c = np.polyfit(grid[-2:], np.array([cRight[-2], cLeft[-1]]), 1)
        a, b = c[0], c[1]
        # a, b = interp_coeff[-1]
        if abs(a)<1e-8:
            if abs(b)<1e-8:
                lambda1 = 1.0
                mu1 = grid[-1]
                y = grid[-1]
                F1 = quad(integrand_linear, grid[-2], grid[-1])[0]
            elif b>0:
                # b = abs(b)
                lambda1 = np.exp(sqrt(b)*grid[-1])
                mu1 = np.exp(-sqrt(b)*grid[-1])
                y = grid[-1]
                F1 = quad(integrand_sinh, grid[-2], grid[-1])[0]
            else:
                b = abs(b)
                lambda1 = np.sin(sqrt(b) * grid[-1])
                mu1 = np.cos(sqrt(b) * grid[-1])
                y = grid[-1]
                F1 = quad(integrand_sin, grid[-2], grid[-1])[0]
        else:
            z1 = q(grid[-2]) * np.power(np.abs(a), -2 / 3)
            z2 = q(grid[-1]) * np.power(np.abs(a), -2 / 3)
            lambda1 = airy(z2)[0]
            mu1 = airy(z2)[2]
            z = z2
            F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
        U[2*N-3, -2:] = np.array([-lambda1, -mu1])
        # hU = np.loadtxt("D:\论文\interface-1d\interface-1d\A.csv", delimiter=',')
        B[2*N-3] = F1
        B[N-2] -= 1.0
        B[N-1] -= 1.0

        AB = np.linalg.solve(U, B).flatten()
        each_x = np.zeros(N)
        for i in range(1, N-1):
            each_x[i] = AB[2*(i-1)]*coeff[i-1][0]+AB[2*(i-1)+1]*coeff[i-1][1]+coeff[i-1][2]
        x[k][:] = each_x
    
        x1[k, :] = x[k, :int((N+1)/2)]
        x2[k, :] = x[k, int((N+1)/2)-1:]
        x2[k, 0] += 1
    return x1, x2



f=generate_data_1d.generate(samples=10)
u1, u2 = discontinuous_tfpm(f)
idx= np.arange(0, 1001)
for i in range(10):
    plt.plot(idx[:501], u1[i].flatten(), label='u1')
    plt.plot(idx[500:], u2[i].flatten(), label='u2')
plt.legend()
plt.savefig('ex3/test.png')
