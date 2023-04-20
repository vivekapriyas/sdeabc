import numpy as np

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
 