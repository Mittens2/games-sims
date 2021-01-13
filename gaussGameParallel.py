import torch
import random
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

# Hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch = 1
trials = 10
eps = 1e-1
n = 5000
eta_1 = 5e-3
eta_2 = 5e-4
h = 10

def G(x, mu):
    return torch.exp(-(x - mu)**2/2)

# Intitialize means of gaussians
mu1 = torch.FloatTensor(batch).uniform_(0, h / 2)
mu2 = torch.FloatTensor(batch).uniform_(h / 2, h)

# Intitialize model parameters
g1_mu = torch.zeros((2, batch), device=device)
g2_mu = torch.zeros((2, batch), device=device)
d_l = torch.zeros((2, batch), device=device)
d_r = torch.zeros((2, batch), device=device)
d_a = torch.zeros((2, batch), device=device)
d_b = torch.zeros((2, batch), device=device)
s = torch.zeros((2, batch), device=device)

success = 0
for t in range(trials):
    # Initialize batch
    g1_mu[0] = torch.FloatTensor(batch).uniform_(0, h)
    g2_mu[0] = torch.FloatTensor(batch).uniform_(0, h)
    s[0] = torch.FloatTensor(batch).uniform_(0, h)
    d_l[0]= torch.FloatTensor(batch).uniform_(0, h / 4)
    d_r[0]= torch.FloatTensor(batch).uniform_(h / 4, h / 2)
    d_a[0]= torch.FloatTensor(batch).uniform_(h / 2, 3 * h / 4)
    d_b[0]= torch.FloatTensor(batch).uniform_(3 * h / 4 , h)
    for i in range(1, n):
        # Update discriminator 1
        d_r[1] = d_r[0] + eta_1 * ((d_r[0] >= d_l[0]).type(torch.float32) * 2 - 1) * (G(d_r[0], mu1) + G(d_r[0], mu2) - G(d_r[0], g1_mu[0]) - G(d_r[0], g2_mu[0]))
        d_l[1] = d_l[0] + eta_1 * ((d_r[0] >= d_l[0]).type(torch.float32) * 2 - 1) * (G(d_l[0], g1_mu[0]) + G(d_l[0], g2_mu[0]) - G(d_l[0], mu1) - G(d_l[0], mu2))
        # Update discriminator 2
        d_b[1] = d_b[0] + eta_1 * ((d_b[0] >= d_a[0]).type(torch.float32) * 2 - 1) * (G(d_b[0], mu1) + G(d_b[0], mu2) - G(d_b[0], g1_mu[0]) - G(d_b[0], g2_mu[0]))
        d_a[1] = d_a[0] + eta_1 * ((d_b[0] >= d_a[0]).type(torch.float32) * 2 - 1) * (G(d_a[0], g1_mu[0]) + G(d_a[0], g2_mu[0]) - G(d_a[0], mu1) - G(d_a[0], mu2))
        # Update Generatorss
        g1_mu[1] = g1_mu[0] - eta_1 * (G(d_r[0], g1_mu[0]) - G(d_l[0], g1_mu[0]) + G(d_b[0], g1_mu[0]) - G(d_a[0], g1_mu[0])) + eta_2 * (g1_mu[0] - s[0]) / (1 + (g1_mu[0] - s[0])) ** 2
        g2_mu[1] = g2_mu[0] - eta_1 * (G(d_b[0], g2_mu[0]) - G(d_a[0], g2_mu[0]) + G(d_r[0], g2_mu[0]) - G(d_l[0], g2_mu[0])) + eta_2 * (g2_mu[0] - s[0]) / (1 + (s[0] - g2_mu[0])) ** 2
        # Update Securer
        s[1] = s[0] - eta_2 * (g1_mu[0] - s[0]) / (1 + (g1_mu[0] - s[0])) ** 2 - eta_2 * (g2_mu[0] - s[0]) / (1 + (s[0] - g2_mu[0])) ** 2
    # Check number of sucesses
    success += torch.sum(torch.min(torch.abs(g1_mu[1] - mu1), torch.abs(g1_mu[1] - mu2)) + torch.min(torch.abs(g2_mu[1] - mu1), torch.abs(g2_mu[1] - mu2)) < eps)

# print('disc: ', d_l[1], d_r[1])
# print('gen means: ', g1_mu[1], g2_mu[1])
# print('true means', mu1, mu2)
print(success / (trials * batch))
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# fig.subplots_adjust(hspace=.5)
# ax2 = fig.add_subplot(212)
# ax1.plot(np.arange(n), g1_mu, label='g1', color='red')
# ax1.plot(np.arange(n), g2_mu, label='g2', color='blue')
# ax1.plot(np.arange(n), d_l, label='dl', linestyle=':', color='blue')
# ax1.plot(np.arange(n), d_r, label='dr', linestyle=':', color='red')
# ax1.plot(np.arange(n), d_a, label='da', linestyle=':', color='blue')
# ax1.plot(np.arange(n), d_b, label='db', linestyle=':', color='red')
# ax1.plot(np.arange(n), mu1 * np.ones(n), linestyle='--', color='black')
# ax1.plot(np.arange(n), mu2 * np.ones(n), linestyle='--', color='black')
# ax1.legend(loc='upper right')
# x = np.arange(0, h, h / 100)
# ax2.plot(x, G(x, g2_mu[n-1]) + G(x, g1_mu[n-1]), label='gen')
# ax2.plot(x, G(x, mu1) + G(x, mu2), label='true')
# ax2.legend(loc='upper right')
# reg_str = 'no' if eta_2==0 else 'yes'
# plt.title('gaussian game, ' + reg_str + ' reg')
# plt.show()
