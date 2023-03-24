import numpy as np

def brownian(t_end = 100, M = 1, N = 10**4):
    """
    returns M discretized Brownian paths on [0, t_end]
    """
    W = np.zeros((N+1, M))
    dt = t_end / (N + 1)
    t = np.linspace(0, t_end, N +1)
    xi = np.random.randn(N, M)
    W[1:] = np.cumsum((xi * np.sqrt(dt)), axis = 0) 
    return (t, W)
