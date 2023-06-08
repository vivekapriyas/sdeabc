import numpy as np

class SDE:
    def __init__(self, x0: float, with_stats = False, correlated_solutions = False) -> None:
        self.x0 = x0
        assert not (with_stats and correlated_solutions), 'with_stats and correlated_solutions not implemented'
        self.solve = self.get_solve(with_stats, correlated_solutions)

    def implicit_milstein(self, x: np.array, dt: float, dW: float) -> np.array:
        raise NotImplementedError
    
    def statistics(self, x: np.array):
        raise NotImplementedError
    
    def set_statsval(self, with_stats: bool) -> None:
        self.solve = self.get_solve(with_stats)
    
    def get_solve(self, with_stats = False, correlated_solutions = False):
        if with_stats:
            def solve(self, M: int, N: int, burn_in = 0) -> np.array:
                dt = 1/(N + 1)
                dW = np.sqrt(dt) * np.random.randn(N)
                x = np.zeros((M, N))
                x[:,0] = self.x0
                for n in range(N - 1):
                    x[:,n + 1] = self.implicit_milstein(x[:,n], dt, dW[n])
                return self.statistics(x)
        
        elif correlated_solutions:
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

    def numerical_solution(self, M: int, N: int, burn_in = 0) -> np.array:
        return self.solve(self, M, N, burn_in)