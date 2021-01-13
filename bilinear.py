import numpy as np
import random
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

# Hyperparameters
# alpha = 0.005 for smooth
# alpha = 0.05 for ogd linear
n = 400
alpha = 0.05
d = 1

# Simple blinear function
def b(x, y):
    return x * y

def db(x, y, i):
    if i == 0:
        return y
    else:
        return x

def da(x, y, i):
    if i == 0:
        if x * y >= 0:
            return y
        else:
            return -y
    else:
        if x * y >= 0:
            return -x
        else:
            return x

# Critical seems to be about 22...
def db3(x, y, z, i):
    if i == 0:
        return A_xz@z + A_xy@y
    elif i == 1:
        return -A_xy.T@x + A_yz@z
    else:
        return -A_xz.T@x - A_yz.T@y

def da3(x, y, z, i):
    if i == 0:
        retval = 0
        if x * y >= 0:
            retval += y
        else:
            retval -= y
        if x * z >= 0:
            retval += z
        else:
            retval -= z
        return retval
    elif i == 1:
        retval = 0
        if x * y >= 0:
            retval -= x
        else:
            retval += x
        if y * z >= 0:
            retval += z
        else:
            retval -= z
        return retval
    else:
        retval = 0
        if x * z >= 0:
            retval -= x
        else:
            retval += x
        if y * z >= 0:
            retval -= y
        else:
            retval += y
        return retval

# Critical is 0.5 (point at which 2P game has incenitve to go towards eq)
def dq3(x, y, z, i):
    if i == 0:
        return -x + A_xy@y + A_xz@z
    elif i == 1:
        return -y - A_xy.T@x + A_yz@z
    else:
        return  -z - A_xz.T@x - A_yz.T@y


# Variationally coherent function
def v(x, y):
    return (x ** 4 * y ** 2 + 2 * x + 1) * (x ** 2 * y ** 4 - y ** 2 + 1)

def dv(x, y, i):
    if i == 0:
        return (4 * x ** 3 * y ** 2 + 2 * x) * (x ** 2 * y ** 4 - y ** 2 + 1) \
                + (2 * x * y ** 4) * (x ** 4 * y ** 2 + x ** 2 + 1)
    else:
        return (4 * y ** 3 * x ** 2 - 2 * y) * (y ** 2 * x ** 4 + x ** 2 + 1) \
                + (2 * y * x ** 4) * (y ** 4 * x ** 2 - y ** 2 + 1)

# Strongly-concave strongly-convex
def q(x, y):
    return - y ** 2 / 2 + y * x + x ** 2 / 2

def dq(x, y, i):
    if i == 0:
        return x / 5 + A_xy@y
    else:
        return -y / 5 + A_xy.T@x

# Non-convex (non-convex/strongly-concave)
def z(x, y):
    return - x ** 2 / 8 + 6 * x * y / 10 - y ** 2 / 2

def dz(x, y, i):
    if i == 0:
        return - x / 4 + 6 * y / 10
    else:
        return - y + 6 * x / 10

# Rosenbrock function, locally coherent
def r(x, y):
    return (1 - x ** 2) + 100 * (y - x ** 2) ** 2

def dr(x, y, i):
    if i == 0:
        return 2 * (1 - x) - 400 * x * (y - x ** 2)
    else:
        return 200 * (y - x ** 2)

# Linear/strongly-convex function
def l(x, y):
    return x * y - y ** 2 / 2

def dl(x, y, i):
    if i == 0:
        return y
    else:
        return x - y

# Gradient Descent Variants
def gd(x, y, df, i):
    x_t = x[i] - alpha * df(x[i], y[i], 0)
    y_t = y[i] + alpha * df(x[i], y[i], 1)
    return x_t, y_t

def gd3(x, y, z, df, i):
    x_t = x[i] + alpha * df(x[i], y[i], z[i], 0)
    y_t = y[i] + alpha * df(x[i], y[i], z[i], 1)
    z_t = z[i] + alpha * df(x[i], y[i], z[i], 2)
    return x_t, y_t, z_t

def pgd(x, y, df, i):
    x_t =  x[i] - alpha * df(x[i], y[i], 0)
    y_t =  y[i] + alpha * df(x_t, y[i], 1)
    return x_t, y_t

# Approximation for both players being prsecient
def ogd(x, y, df, i):
    x_t =  x[i] - 2 * alpha * df(x[i], y[i], 0) + alpha * df(x[i-1], y[i-1], 0)
    y_t =  y[i] + 2 * alpha * df(x[i], y[i], 1) - alpha * df(x[i-1], y[i-1], 1)
    return x_t, y_t

def ogd3(x, y, z, df, i):
    x_t =  x[i] + 2 * alpha * df(x[i], y[i], z[i], 0) - alpha * df(x[i-1], y[i-1], z[i-1], 0)
    y_t =  y[i] + 2 * alpha * df(x[i], y[i], z[i], 1) - alpha * df(x[i-1], y[i-1], z[i-1], 1)
    z_t =  z[i] + 2 * alpha * df(x[i], y[i], z[i], 2) - alpha * df(x[i-1], y[i-1], z[i-1], 2)
    return x_t, y_t, z_t

def pogd(x, y, df, i):
    x_t =  x[i] - 2 * alpha * df(x[i], y[i], 0) + alpha * df(x[i-1], y[i-1], 0)
    y_t =  y[i] + alpha * df(x_t, y[i], 1)
    return x_t, y_t

def gdbr(x, y, df, i):
    x_t = x[i] - alpha * df(x[i], y[i], 0)
    y_t = (x[i] >= 0) * 2 - 1
    return x_t, y_t

def ogdbr(x, y, df, i):
    x_t =  x[i] - 2 * alpha * df(x[i], y[i], 0) + alpha * df(x[i-1], y[i-1], 0)
    y_t = (x_t >= 0) * 2 - 1
    return x_t, y_t

# Averaging
def avg(x):
    x_bar = np.zeros(x.shape)
    for i in range(len(x)):
        x_bar[i] = (x[i] + x_bar[i - 1] * i) / (i + 1)
    return x_bar

# Define dynamics
# A_xy = np.random.rand(d,d)
# A_xz = np.random.rand(d,d)
# A_yz = np.random.rand(d,d)
A_xy = np.random.normal(size=(d,d))
A_xz = np.random.normal(size=(d,d))
A_yz = np.random.normal(size=(d,d))
# A_xy = np.ones((1,1))
# A_xz = np.ones((1,1))
# A_yz = np.ones((1,1))
A = np.block([[np.zeros((d,d)), A_xy, A_xz],[-A_xy.T, np.zeros((d,d)), A_yz],[-A_xz.T, -A_yz.T, np.zeros((d,d))]])
print(np.linalg.det(A))
c_x = np.random.uniform()
c_y = np.random.uniform()
c_z = np.random.uniform()
f, df = q, dq
opt = [gd, ogd]
names = ['gd', 'ogd']
fig = plt.figure()
ax1 = fig.add_subplot(111)
# ax1 = fig.add_subplot(211, projection='3d')
fig.subplots_adjust(hspace=.5)
# ax2 = fig.add_subplot(212)
x = np.zeros((n, d))
y = np.zeros((n, d))
z = np.zeros((n, d))
x[0] = np.random.uniform(low=-3, high=3, size=d)
y[0] = np.random.uniform(low=-3, high=3, size=d)
z[0] = np.random.uniform(low=-3, high=3, size=d)
x[1] = x[0]
y[1] = y[0]
z[1] = z[0]
for g, name in zip(opt, names):
    for i in range(1, n - 1):
        # x_t, y_t, z_t = g(x, y, z, df, i)
        x_t, y_t = g(x, y, df, i)
        x[i + 1] = x_t
        y[i + 1] = y_t
        # z[i + 1] = z_t
        # print(x[i + 1], y[i + 1], z[i + 1])
    x_bar, y_bar, z_bar = avg(x), avg(y), avg(z)
    # x_plot, y_plot, z_plot = np.sum(x[1:] ** 2, axis=1), np.sum(y[1:] ** 2, axis=1), np.sum(z[1:] ** 2, axis=1)
    # ax1.plot(xs=x_plot, ys=y_plot, zs=z_plot, label='last')
    # ax1.plot(xs=np.sum(x_bar[1:] ** 2, axis=1), ys=np.sum(y_bar[1:] ** 2, axis=1), zs=np.sum(z_bar[1:] ** 2, axis=1), label='avg', color='orange')
    ax1.plot(x[1:], y[1:], label=name)
    # ax2.plot(np.arange(n-1), np.sum(x[1:] ** 2, axis=1) + np.sum(y[1:] ** 2, axis=1) + np.sum(z[1:] ** 2, axis=1), label=name)
    # ax2.plot(np.arange(n-1), np.sum(x_bar[1:] ** 2, axis=1) + np.sum(y_bar[1:] ** 2, axis=1) + np.sum(z_bar[1:] ** 2, axis=1), label='avg')

    # ax2.plot(np.arange(n), f(x, y, z), label=name)

# 2d plot
ax1.plot(x[1, 0], y[1, 0], marker='o')
ax1.plot(0, 0, marker='o')
delta = 0.025
x = np.arange(-4, 4, delta)
y = np.arange(-4 , 4, delta)
X, Y = np.meshgrid(x, y)
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
CS = ax1.contour(X, Y, f(X, Y), 10, colors='k', alpha=0.2)
ax1.clabel(CS, inline=1, fontsize=10)
ax1.legend(loc='upper right')
#3d plots
# ax1.locator_params(nbins=5)
# ax1.title.set_text('3D trajectory')
# ax1.legend(loc='upper left')
# ax1.set_xlabel('x1', labelpad=10)
# ax1.set_ylabel('x2')
# ax1.set_zlabel('x3')
# ax2.legend(loc='upper right')
# ax2.title.set_text('Distance to Equillibrium')
# ax2.set_xlabel('iterates')
# ax2.set_ylabel('sum L2')

plt.show()
