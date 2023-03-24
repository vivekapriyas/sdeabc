import numpy as np
from scipy.stats import norm

class Proposal:
    def __init__(self) -> None:
        pass

    def density(self, x, y):
        pass

    def step(self, x):
        pass

class RandomWalk(Proposal):
    def __init__(self, sigma) -> None:
        self.sigma = sigma
        super().__init__()

    def density(self, x, y):
        return norm.pdf(y, loc = x, scale = self.sigma)
    
    def step(self, x):
        return x + self.sigma * np.random.normal(size = x.shape())