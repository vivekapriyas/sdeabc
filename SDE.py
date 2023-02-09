import numpy as np

class SDE(object):
    def __init__(self, f, g, x0):
        self.f = f
        self.g = g
        self.x0 = x0

    def exact_solution(self):
        raise NotImplementedError
    


class linear_SDE(SDE):
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