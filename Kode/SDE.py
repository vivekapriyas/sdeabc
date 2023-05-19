import numpy as np

class SDE:
    def __init__(self, x0: float) -> None:
        self.x0 = x0

    def implicit_milstein(self, x: np.array, dt: float, dW: float) -> np.array:
        raise NotImplementedError
    
    def get_solve(self, correlated_solutions: bool):
        if correlated_solutions:
            def solve(self, M: int, N: int, burn_in = 0) -> np.array:
                dt = 1/(N + 1)
                dW = np.sqrt(dt) * np.random.randn(N)
                x = np.zeros((M, N))
                y = np.zeros((M, N))
                x[:,0] = self.x0
                y[:,0] = self.x0
                for n in range(N - 1):
                    x[:,n + 1] = self.implicit_milstein(x[:,n], dt, dW[n])
                    y[:,n + 1] = self.implicit_milstein(y[:,n], dt, -dW[n])
                return np.array([x.ravel(order = 'F')[burn_in:], y.ravel(order = 'F')[burn_in:]]) 

        else:
            def solve(self, M: int, N: int, burn_in = 0) -> np.array:
                dt = 1/(N + 1)
                dW = np.sqrt(dt) * np.random.randn(N)
                x = np.zeros((M, N))
                x[:,0] = self.x0
                for n in range(N - 1):
                    x[:,n + 1] = self.implicit_milstein(x[:,n], dt, dW[n])
                return x.ravel(order = 'F')[burn_in:]
        return solve

    def numerical_solution(self, M: int, N: int, burn_in = 0, correlated_solutions = False) -> np.array:
        solve = self.get_solve(correlated_solutions)
        return solve(self, M, N, burn_in)