import numpy as np
import random
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from scipy import special

# Hyperparameters
n = 50000
eta_1 = 0.001
eta_2 = 0.0
h = 8

# Constrained game
def f(x):
    return max(min(x, h), 0)

def erf(r, l, mu):
    return math.sqrt(math.pi) / 2 * (special.erf((r - mu) / math.sqrt(2)) - special.erf((l - mu) / math.sqrt(2)))

def G(x, mu):
    return np.exp(-(x - mu)**2/2)

def logistic(x):
    return 1 / (1 + np.exp(-x))

# Intitialize model parameters
g1_mu = np.zeros(n)
g2_mu = np.zeros(n)
d_l = np.zeros(n)
d_r = np.zeros(n)
d_a = np.zeros(n)
d_b = np.zeros(n)

trials = 50
tv = 0
# success = 0
# eps = 1e-1

for t in range(trials):
    mu1 = np.random.uniform(high=h/4)
    mu2 = mu1 + h/2
    g1_mu[0] = np.random.uniform(high=h/2)
    g2_mu[0] = np.random.uniform(low=h/2, high=h)
    d_l[0] = np.random.uniform(high=h/4)
    d_r[0] = np.random.uniform(low=h/4, high=h/2)
    d_a[0] = np.random.uniform(low=h/2, high=3*h/4)
    d_b[0] = np.random.uniform(low=3*h/4, high=h)
    for i in range(n - 1):
        # Update discriminator 1
        d_r[i+1] = f(d_r[i] + eta_1 * (int(d_r[i] >= d_l[i]) * 2 - 1) * (G(d_r[i], mu1) + G(d_r[i], mu2) - G(d_r[i], g1_mu[i]) - G(d_r[i], g2_mu[i])))
        d_l[i+1] = f(d_l[i] + eta_1 * (int(d_r[i] >= d_l[i]) * 2 - 1) * (G(d_l[i], g1_mu[i]) + G(d_l[i], g2_mu[i]) - G(d_l[i], mu1) - G(d_l[i], mu2)))
        # Update discriminator 2
        d_b[i+1] = f(d_b[i] + eta_1 * (int(d_b[i] >= d_a[i]) * 2 - 1) * (G(d_b[i], mu1) + G(d_b[i], mu2) - G(d_b[i], g1_mu[i]) - G(d_b[i], g2_mu[i])))
        d_a[i+1] = f(d_a[i] + eta_1 * (int(d_b[i] >= d_a[i]) * 2 - 1) * (G(d_a[i], g1_mu[i]) + G(d_a[i], g2_mu[i]) - G(d_a[i], mu1) - G(d_a[i], mu2)))
        # Update Generators
        g1_mu[i+1] = f(g1_mu[i] - eta_1 * (G(d_r[i], g1_mu[i]) - G(d_l[i], g1_mu[i]) + G(d_b[i], g1_mu[i]) - G(d_a[i], g1_mu[i])))
        g2_mu[i+1] = f(g2_mu[i] - eta_1 * (G(d_b[i], g2_mu[i]) - G(d_a[i], g2_mu[i]) + G(d_r[i], g2_mu[i]) - G(d_l[i], g2_mu[i])))

    g1_dist = np.array([abs(g1_mu[n-1] - mu1), abs(g1_mu[n-1] - mu2)])
    g2_dist = np.array([abs(g2_mu[n-1] - mu1), abs(g2_mu[n-1] - mu2)])
    ind = np.argmin(g1_dist)
    tv += g1_dist[ind] + g2_dist[1 - ind]
    # if g1_dist[ind] + g2_dist[1 - ind] < eps:
    #     success += 1

print('disc: ', d_l[n-1], d_r[n-1])
print('gen means: ', g1_mu[n-1], g2_mu[n-1])
print('true means', mu1, mu2)
# print(success / trials)
print(tv)
fig = plt.figure()
ax1 = fig.add_subplot(211)
fig.subplots_adjust(hspace=.5)
ax2 = fig.add_subplot(212)
ax1.plot(np.arange(n), g1_mu, label='g1', color='red')
ax1.plot(np.arange(n), g2_mu, label='g2', color='blue')
# ax1.plot(np.arange(n), s, label='s', color='green')
ax1.plot(np.arange(n), d_l, label='dl', linestyle=':', color='blue')
ax1.plot(np.arange(n), d_r, label='dr', linestyle=':', color='red')
ax1.plot(np.arange(n), d_a, label='da', linestyle=':', color='blue')
ax1.plot(np.arange(n), d_b, label='db', linestyle=':', color='red')
ax1.plot(np.arange(n), mu1 * np.ones(n), linestyle='--', color='black')
ax1.plot(np.arange(n), mu2 * np.ones(n), linestyle='--', color='black')
ax1.legend(loc='upper right')
x = np.arange(0, h, h / 100)
ax2.plot(x, G(x, g2_mu[n-1]) + G(x, g1_mu[n-1]), label='gen')
ax2.plot(x, G(x, mu1) + G(x, mu2), label='true')
ax2.legend(loc='upper right')
reg_str = 'no' if eta_2==0 else 'yes'
plt.title('gaussian game, ' + reg_str + ' reg')
plt.show()
