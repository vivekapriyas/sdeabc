import numpy as np

def stationary_statistics(X):
    return np.array([np.mean(X), np.var(X), np.corrcoef(X[:2])])