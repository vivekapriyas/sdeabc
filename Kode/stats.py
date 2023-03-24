import numpy as np

def stationary_statistics(X):
    return np.mean(X), np.var(X), np.cov(X[:1])