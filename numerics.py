import numpy as np

def BM(t_end = 1, M = 1, N = 10**4):
    """
    returns a discretized Brownian path on [0, t_end]
    """
    np.random.seed(1234)
    W = np.zeros((N+1, M))
    dt = t_end / (N + 1)
    t = np.linspace(0, t_end, N +1)
    xi = np.random.randn(N, M)
    W[1:] = np.cumsum((xi * np.sqrt(dt)), axis = 0) 
    
    return (t, W)


def euler_maruyama(Y, t, x, dt, dW):
    """
    returns one euler-maruyama step
    """
    return x + Y.f(t, x) * dt + Y.g(t, x) * dW 


def num_solution(Y, t, W, method = euler_maruyama):
    """
    returns numerical solution of SDE Y on [0, t_end]
    """
    N, M = np.shape(W)
    X = np.zeros((N, M))
    dt = t[1]-t[0]
    
    X[0,:] = Y.x0
    for n in range(N-1):
        dW = W[n + 1,:] - W[n,:]
        X[n + 1,:] = method(Y, t[n], X[n,:], dt, dW)
    return X