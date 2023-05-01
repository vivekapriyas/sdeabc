import numpy as np
from scipy.special import gamma

class Distance:
    def __init__(self) -> None:
        pass

    def dist(self, x, y) -> np.array:
        raise NotImplementedError
        

class EuclidianDistance(Distance):
    def __init__(self) -> None:
        super().__init__()

    def dist(self, x, y) -> np.array:
        """
        x: n x q array
        y: 1 x q array
        returns: 1 x n array 
        """
        return np.array([np.linalg.norm((i - y)) for i in x])
 

class KernelDistance(Distance):
    def __init__(self, p) -> None:
        self.p = p
        self.V = self.Vp(p)

    def Vp(self, p):
        return (1 / np.pi) * (gamma(p / 2) * p/2)**(p/2)
    
    def cp(self, A):
        V, p = self.V, self.p
        return V * np.linalg.det(A)**(1/p)
    
    def dist(self, x, y):
        pass