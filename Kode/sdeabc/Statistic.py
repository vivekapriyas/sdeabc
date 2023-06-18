import numpy as np
from statsmodels.tsa.stattools import acf
import time

def ac(x: np.array, lag = 1) -> int:
    n = len(x)
    mn = np.mean(x)
    v = np.var(x)
    return ((1 / n)* np.sum(x[:(n - lag)] * x[lag:]) - mn**2) / v

class Statistic:
    def __init__(self) -> None:
        pass
    
    def get_dim(self) -> int:
        raise NotImplementedError

    def statistic(self, x) -> np.array:
        raise NotImplementedError
    

class Identity(Statistic):
    def __init__(self, dim) -> None:
        self.dim = dim
        super().__init__()
    
    def get_dim(self) -> int:
        return self.dim

    def statistic(self, x) -> np.array:
        d = self.get_dim()
        return np.reshape(x, (d))

class Autocov(Statistic):
    """
    first q autocovariances
    """
    def __init__(self, q = 2) -> None:
        self.q = q
        super().__init__()
    
    def get_dim(self) -> int:
        return self.q
    
    def statistic(self, x: np.array) -> np.array:
        """
        x: n x k array
        returns:n x q array
        """
        return np.array([acf(i, nlags = self.q)[1:] for i in x])

class StationaryStats(Statistic):
    def __init__(self) -> None:
        super().__init__()

    def get_dim(self) -> int:
        return 3

    def statistic(self, x: np.array) -> np.array:
        """
        x: n x k array
        returns: n x 3 array
        """
        d = self.get_dim()
        m = np.mean(x)
        sd = np.std(x)
        c = ac(x, lag = 1)
        return np.reshape(np.array([c, m, sd]), (d))