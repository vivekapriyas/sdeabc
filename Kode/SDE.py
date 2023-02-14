import numpy as np
from scipy.special import gamma, gammainc
from numerics import *

class SDE(object):
    def __init__(self, f, g, x0) -> None:
        self.f = f
        self.g = g
        self.x0 = x0
        pass

    def exact_solution(self):
        raise NotImplementedError
    


class Linear_SDE(SDE):
    '''
    dX = (mu * X)dt + (sigma * X)dB
    '''
    def __init__(self, x0, mu = 1, sigma = 0.5):       
        self.mu, self.sigma = mu, sigma
        f = lambda t, x : mu * x 
        g = lambda t, x : sigma * x
        self.L1g = lambda t, x: sigma**2 * x
        self.eq = f'dXt = {self.mu}Xdt + {self.sigma}XdW'
        super().__init__(f, g, x0)
        
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
    

class Gamma_SDE(SDE):
    """
    SDE with two parameter Gamma stationary distribution (shape, scale)
    and exponentially decaying autocorrelation c(l) = e^-ac*l
    """
    def __init__(self, shape, scale, ac, x0) -> None:
        self.shape, self.scale, self.ac = shape, scale, ac
        f = lambda t, x: - self.ac * (x - self.shape * self.scale)
        g = lambda t, x: np.sqrt(2 * self.ac * self.scale * x)
        super().__init__(f, g, x0)


class Weibull_SDE(SDE):

    def __init__(self, shape, scale, ac, x0) -> None:
        self.shape, self.scale, self.ac = shape, scale, ac
        f = lambda t, x: - self.ac * (x - self.scale * gamma(1 + 1/shape))
        b1 = lambda x: 2 * self.ac * scale**(shape + 1) / shape**2 * x**(1 - shape)
        b2 = lambda x: shape * np.exp((x / scale)**shape) * gammainc(1 + 1/shape, (x / scale)**shape) - gamma(1 / shape)
        g = lambda t, x: np.sqrt(b1(x) * b2(x))
        super().__init__(f, g, x0)