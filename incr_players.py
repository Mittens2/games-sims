import numpy as np
import random
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

# Hyperparameters
n = 20000
alpha = 0.001
d = 1
thresh = 1e-5
max_p = 100
trials = 10

def gd(C, x, i):
    return -(x[i] - C@x[i])

def ogd(C, x, i):
    return -2 * (x[i] - C@x[i]) + (x[i-1] - C@x[i-1])

p_iterates = np.zeros((2, max_p - 3))
opt = [gd, ogd]
for p in range(3, max_p):
    x = np.zeros((n + 1, p * d))
    x[0:2] = np.random.rand(2, p * d)
    iu = np.tril_indices(n=p * d, m=p * d)
    C = np.random.rand(p * d, p * d)
    C = (C + C.T) / 2 # random symmetric matrix
    C[iu] *= -1 # skew symmetric
    # Set diagonal blocks to 0
    for i in range(p):
        C[i * d: (i + 1) * d, i * d: (i + 1) * d] = 0
    for j, g in enumerate(opt):
        for t in range(trials):
            for i in range(1, n):
                x[i + 1] = x[i] + alpha * g(C, x, i)
                print(np.sum(x[i + 1] ** 2))
                if np.sum(x[i + 1] ** 2) < thresh:
                    p_iterates[j, p - 3] += i
                    break
        p_iterates[j, p -3] /= trials

plt.plot(np.arange(3, max_p), p_iterates[0], label='GA')
plt.plot(np.arange(3, max_p), p_iterates[1], label='OGA')
plt.xlabel('players')
plt.ylabel('iterates')
plt.legend(loc='upper left')
plt.show()
