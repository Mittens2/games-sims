import numpy as np
import random
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

# Hyperparameters
n = 20000
batch = 200
eta = 1e-3
beta = 1e-4
h = 1

# Projection onto space
def p(x):
    return max(min(x, h), 0)

# Quadratic loss
def dqd(w, x, z):
    return (z - x)

def dqg(b, w, z):
    return (w - z)

# Logistic loss
def ded(w, x, z):
    return z / (np.exp(-w * z) + 1) - x / (np.exp(w * x) + 1)

def deg(b, w, z):
    return w / (np.exp(- w * z) + 1)

# Intitialize true distribution
sep = 0.2
mu1 = np.random.uniform(high=h/2)
mu2 = mu1 + sep

# Intitialize model parameters
d_w1 = np.zeros(n)
d_w2 = np.zeros(n)
g_b1 = np.zeros(n)
g_b2 = np.zeros(n)

g_b1[0] = np.random.uniform(high=h)
g_b2[0] = np.random.uniform(high=h)
d_w1[0] = np.random.uniform(high=h/2)
d_w2[0] = np.random.uniform(low=h/2, high=h)

dd = ded
dg = deg
for i in range(0, n-1):
    # Sample from true distribution
    mask = (np.random.uniform(size=batch) >= 0.5)
    mask_sz = mask[mask == 0].shape[0]
    x = np.zeros(batch)
    x[mask == 1] = np.random.normal(loc=mu1, size=(batch - mask_sz))
    x[mask == 0] = np.random.normal(loc=mu2, size=mask_sz)
    # Sample from each generator
    z_1 = np.random.normal(size=batch) + g_b1[i]
    z_2 = np.random.normal(size=batch) + g_b2[i]
    # Update parameters using GDA
    d_w1[i+1] = d_w1[i] - eta * np.sum(dd(d_w1[i], x, z_1) + dd(d_w1[i], x, z_2)) / batch #+ beta * (d_w1[i] - d_w2[i])
    d_w2[i+1] = d_w2[i] - eta * np.sum(dd(d_w2[i], x, z_1) + dd(d_w2[i], x, z_2)) / batch #+ beta * (d_w2[i] - d_w1[i])
    g_b1[i+1] = g_b1[i] + eta * np.sum(dg(g_b1[i], d_w1[i], z_1) + dg(g_b1[i], d_w2[i], z_1)) / batch #+ beta * (g_b1[i] - g_b2[i])
    g_b2[i+1] = g_b2[i] + eta * np.sum(dg(g_b2[i], d_w1[i], z_2) + dg(g_b2[i], d_w2[i], z_2)) / batch #+ beta * (g_b2[i] - g_b1[i])
    # print(g_b1[i+1] - g_b2[i+1])

# print('disc: ', d_w1[n-1], d_w2[n-1])
print('gen means: ', g_b1[n-1], g_b2[n-1])
print('true means', mu1, mu2)
plt.plot(np.arange(n), g_b1, label='g1', color='red')
plt.plot(np.arange(n), g_b2, label='g2', color='blue')
# plt.plot(np.arange(n), d_w1, label='dl', linestyle=':', color='black')
# plt.plot(np.arange(n), d_w2, label='dr', linestyle=':', color='black')
plt.plot(np.arange(n), mu1 * np.ones(n), linestyle='--', color='black')
plt.plot(np.arange(n), mu2 * np.ones(n), linestyle='--', color='black')
plt.legend(loc='upper right')
plt.title('gaussian game')
plt.show()
