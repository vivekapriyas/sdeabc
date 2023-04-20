import numpy as np
from scipy.optimize import fsolve
from scipy.special import digamma

def stationary_statistics(X):
    return np.array([np.mean(X), np.var(X), np.corrcoef(X[:2])])


def MLE(x):
    n = len(x)
    xbar = np.mean(x)
    a0 = n * (xbar**2) / np.sum((x - xbar)**2)
    alpha = fsolve(lambda a: (np.sum(np.log(x)) - n * np.log(xbar) + n*np.log(a))/n - digamma(a), a0)
    beta = (1/alpha) * xbar
    return alpha[0], beta[0]
