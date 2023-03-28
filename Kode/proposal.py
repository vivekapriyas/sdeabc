import numpy as np
from scipy.stats import uniform, truncnorm,  gamma

class Density:
    def __init__(self) -> None:
        pass

    def density(self, x, y):
        pass

    def step(self, x):
        pass

class RandomWalk(Density):
    def __init__(self, sigma) -> None:
        self.sigma = sigma
        super().__init__()

    def density(self, x, y):
        d = truncnorm.pdf(y, loc = x, a = 0, b = np.inf, scale = self.sigma)
        return d
    
    def step(self, x):
        return truncnorm.rvs(loc = x, a = 0, b = np.inf, size = x.shape)
    

class Prior_gammapar(Density):
    def __init__(self) -> None:
        self.aa, self.ab = 9, 2
        self.ba, self.bb = 2, 2
        super().__init__()

    def density(self, x):
        alpha_d = uniform.pdf(x[0])
        a_d = gamma.pdf(x[1], a = self.aa, scale = 1/self.ab)
        b_d = gamma.pdf(x[2], a = self.ba, scale = 1/self.bb )
        return np.array([alpha_d, a_d, b_d])
    
    def step(self):
        alpha = uniform.rvs(size = 1)[0]
        a = gamma.rvs(a = self.aa, scale = 1 / self.ab, size = 1)[0]
        b = gamma.rvs(a = self.ba, scale = 1 /self.bb, size = 1)[0]
        return np.array([alpha, a, b])
    