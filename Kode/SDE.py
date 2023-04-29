import numpy as np

class SDE:
    def __init__(self, x0: float) -> None:
        self.x0 = x0

    def implicit_milstein(self, x: np.array, dt: float, dW: float) -> np.array:
        raise NotImplementedError
    
    def numerical_solution(self, M: int, N: int, burn_in = 0) -> np.array:
        dt = 1/(N + 1)
        dW = np.sqrt(dt) * np.random.randn(N)
        x = np.zeros((M, N))
        x[:,0] = self.x0
        for n in range(N - 1):
            x[:,n + 1] = self.implicit_milstein(x[:,n], dt, dW[n])
        return x.ravel(order = 'F')[burn_in:] #OBS: vurder bruk av ravel vs reshape