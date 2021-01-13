import numpy as np
import random
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

# Hyperparameters
# alpha = 0.005 for smooth
# alpha = 0.05 for ogd linear
n = 2000
alpha = 0.1
d = 2

# Simple blinear function
def b(x):
    return x * y

def db(x, y, i):
    if i == 0:
        return -A.T@y
    else:
        return A@x

# Gradient Descent Variants
def gd(x, y, df, i):
    x_t = x[i] + alpha * df(x[i], y[i], 0)
    y_t = y[i] + alpha * df(x[i], y[i], 1)
    return x_t, y_t

def mwu(x, y, df, i):
    x_t = x[i] * np.exp(alpha * df(x[i], y[i], 0))
    x_t = x_t / np.sum(x_t)
    y_t = y[i] * np.exp(alpha * df(x[i], y[i], 1))
    y_t = y_t / np.sum(y_t)
    return x_t, y_t

def omwu(x, y, df, i):
    x_t = x[i] * np.exp(- 2 * alpha * df(x[i], y[i], 0) + alpha * df(x[i-1], y[i-1], 0))
    x_t = x_t / np.sum(x_t)
    y_t = y[i] * np.exp(- 2 * alpha * df(x[i], y[i], 1) + alpha * df(x[i-1], y[i-1], 1))
    y_t = y_t / np.sum(y_t)
    return x_t, y_t

def umwu(x, y, df, i):
    x_t = x[i] * np.exp(alpha * df(x[i], y[i], 0))
    y_t = y[i] * np.exp(alpha * df(x[i], y[i], 1))
    return x_t, y_t

def uomwu(x, y, df, i):
    x_t = x[i] * np.exp(- 2 * alpha * df(x[i], y[i], 0) + alpha * df(x[i-1], y[i-1], 0))
    y_t = y[i] * np.exp(- 2 * alpha * df(x[i], y[i], 1) + alpha * df(x[i-1], y[i-1], 1))
    return x_t, y_t

# Averaging
def avg(x):
    x_bar = np.zeros(x.shape)
    for i in range(len(x)):
        x_bar[i] = (x[i] + x_bar[i - 1] * i) / (i + 1)
    return x_bar

# Define dynamics
A = np.array([[1, -1], [-1, 1]])
print(A)
print(A.shape)
f, df = b, db
opt = [omwu, uomwu, mwu]
names = ['omwu', 'uomwu', 'mwu']
x = np.zeros((n, d))
y = np.zeros((n, d))
z = np.zeros((n, d))
x[0] = np.random.uniform(low=0, high=1, size=d)
x[0] = x[0] / np.sum(x[0])
y[0] = np.random.uniform(low=0, high=1, size=d)
y[0] = y[0] / np.sum(y[0])
x[1] = x[0]
y[1] = y[0]
for g, name in zip(opt, names):
    for i in range(0, n - 1):
        x_t, y_t = g(x, y, df, i)
        x[i + 1] = x_t
        y[i + 1] = y_t
        print(x[i + 1], y[i + 1])
    if name[0] == 'u':
        x = x / np.sum(x, axis=1)[:,None]
        y = y / np.sum(y, axis=1)[:,None]
    x_bar, y_bar = avg(x), avg(y)
    plt.plot(x[1:, 0], y[1:, 0], label=name)
    # plt.plot(x_bar[1:, 0], y_bar[1:, 0], label='avg', color='orange')

    # ax2.plot(np.arange(n), f(x, y, z), label=name)

# delta = 0.025
# x = np.arange(-5, 5, delta)
# y = np.arange(-5, 5, delta)
# X, Y = np.meshgrid(x, y)
# matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
# CS = ax1.contour(X, Y, f(X, Y), 10, colors='k', alpha=0.2)
# ax1.clabel(CS, inline=1, fontsize=10)
plt.legend(loc='upper right')
# ax1.locator_params(nbins=5)
# ax1.title.set_text('2D trajectory')
# ax1.legend(loc='upper left')
# ax1.set_xlabel('x1', labelpad=10)
# ax1.set_ylabel('x2')
# ax2.legend(loc='upper right')
# ax2.title.set_text('Distance to Equillibrium')
# ax2.set_xlabel('iterates')
# ax2.set_ylabel('sum L2')

plt.show()
