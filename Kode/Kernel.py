import numpy as np
from scipy.stats import multivariate_normal as mvnr

class Kernel:
    def __init__(self, dim) -> None:
        self.dim = dim
        pass

    def get_dim(self) -> int:
        return self.dim
    
    def step(self, x: np.array) -> np.array:
        raise NotImplementedError
    
    def pdf(self, x: np.array, y: np.array) -> np.array:
        raise NotImplementedError
    

class RandomWalk(Kernel):
    def __init__(self, dim) -> None:
        super().__init__(dim)

    def step(self, x: np.array) -> np.array:
        return mvnr.rvs(mean = x, cov =  np.diag(0.1 * np.ones(len(x))))
    
    def pdf(self, x: np.array, y: np.array) -> np.array:
        """
        OBS: tenk gjennom her, vil kunne tilpasse cov 
        """
        return mvnr.pdf(y, mu = x, cov = np.diag(0.1 * np.ones(len(x))))
