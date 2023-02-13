import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numerics import *

class SDE(object):
    def __init__(self, f, g, x0) -> None:
        self.f = f
        self.g = g
        self.x0 = x0
        pass

    def exact_solution(self):
        raise NotImplementedError
    


class LinSDE(SDE):
    '''
    dX = (mu * X)dt + (sigma * X)dB
    '''
    def __init__(self, x0, mu = 1, sigma = 0.5):       
        self.mu, self.sigma = mu, sigma
        self.f = lambda t, x : mu * x 
        self.g = lambda t, x : sigma * x
        self.L1g = lambda t, x: sigma**2 * x
        self.x0 = x0
        self.eq = f'dXt = {self.mu}Xdt + {self.sigma}XdW'
        
    def exact_solution(self, t, W):
        mu = self.mu
        sigma = self.sigma
        x0 = self.x0
        M = W.shape[1] 
        t_ext = np.outer(t, np.ones(M))
        a = (mu - 0.5 * sigma**2) * t_ext + sigma * W
        ex = np.exp(a) * x0
        return ex
    
    def expectation(self, t):
        mu, x0 = self.mu, self.x0        
        return x0 * np.exp(mu * t)
    

class G2SDE(SDE):
    """
    SDE with two parameter Gamma stationary distribution (shape, scale)
    and exponentially decaying autocorrelation c(l) = e^-ac*l
    """
    def __init__(self, shape, scale, ac, x0) -> None:
        self.shape, self.scale, self.ac = shape, scale, ac
        self.x0 = x0
        self.f = lambda t, x: - self.ac * (x - self.shape * self.scale)
        self.g = lambda t, x: np.sqrt(2 * self.ac * self.scale * x)
        pass

test = G2SDE(2.064, 1.411, 0.25, 1.7)
t, W = BM(t_end = 100, N = 10**6)
X = num_solution(test, t, W)

pd.plotting.autocorrelation_plot(X)
plt.show()