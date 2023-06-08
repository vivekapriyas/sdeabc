import numpy as np
from statsmodels.tsa.stattools import acf
import time

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
        return 3 #NB: legg til acf igjen etter hvert

    def statistic(self, x: np.array) -> np.array:
        """
        x: n x k array
        returns: n x 3 array
        """
        d = self.get_dim()
        m = np.mean(x, axis = 1)
        sd = np.std(x, axis = 1)
        starttime = time.time()
        c = np.array([acf(i, nlags = 1)[1] for i in x])
        print('c takes', time.time() -starttime)
        return np.reshape(np.array([m, sd, c]), (d))