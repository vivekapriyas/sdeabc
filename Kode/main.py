import numpy as np
import sde
from matplotlib import pyplot as plt
from numerics import brownian
from scipy.optimize import fsolve
from scipy.special import digamma


x0, alpha, a, b = 4.5, 0.25, 4.38, 1.06
T = 100
parameters = np.array([alpha * T, a, b])
#M = sde.GammaSDE(x0, parameters)

#t, W = brownian(M = 10**3, N = 10**5)
#X = M.num_solver(t, W)
#np.save('Kode\pg2', X)

X = np.load('Kode\pg2.npy')
x = np.ravel(X, order ='F')
"""e = np.cumsum(x) / np.arange(1, len(x) + 1)
plt.plot(e)
plt.show()"""

def MLE(x):
    n = len(x)
    xbar = np.mean(x)
    a0 = n * (xbar**2) / np.sum((x - xbar)**2)
    alpha = fsolve(lambda a: (np.sum(np.log(x)) - n * np.log(xbar) + n*np.log(a))/n - digamma(a), a0)
    beta = (1/alpha) * xbar
    return alpha[0], beta[0]

a_est, b_est = MLE(x)
alpha_est = np.cov(x[:1])
print(a_est, b_est, alpha_est)