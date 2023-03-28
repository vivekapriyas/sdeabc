import numpy as np
import sde
from matplotlib import pyplot as plt
from numerics import brownian
from scipy.optimize import fsolve
from scipy.special import digamma
from scipy.stats import gamma


x0, alpha, a, b = 4.5, 0.25, 4.38, 1.06
T = 100
parameters = np.array([alpha * T, a, b])
M = sde.GammaSDE(x0, parameters)

t, W = brownian(M = 10**3, N = 10**5)
X = M.num_solver(t, W)
x = np.ravel(X, order ='F')[(2 * 10**7):] 
np.save('Kode\pg2', x)

#X = np.load('Kode\pg2.npy')


#e = np.cumsum(x) / np.arange(1, len(x) + 1)

def MLE(x):
    n = len(x)
    xbar = np.mean(x)
    a0 = n * (xbar**2) / np.sum((x - xbar)**2)
    alpha = fsolve(lambda a: (np.sum(np.log(x)) - n * np.log(xbar) + n*np.log(a))/n - digamma(a), a0)
    beta = (1/alpha) * xbar
    return alpha[0], beta[0]

a_est, b_est = MLE(x)
alpha_est = np.corrcoef(x[:2])
print(a_est, b_est, alpha_est)

#e = np.cumsum(x) / np.arange(1, len(x) + 1)
y = np.linspace(0, 12, 100)
plt.hist(x, bins = 30, edgecolor = 'tab:blue', color='steelblue', alpha = .7, density = True)
plt.plot(y, gamma.pdf(y,a = a, scale = 1/b), label = 'original par')
plt.plot(y, gamma.pdf(y, a = a_est, scale = 1 / b_est), label = 'est. par')
plt.legend('topright')
plt.show()