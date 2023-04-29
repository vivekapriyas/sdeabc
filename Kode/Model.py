import numpy as np
import SDE

class Model:
    def __init__(self) -> None:
        pass

    def simulate(self) -> np.array:
        raise NotImplementedError

    def pdf(self, x) -> np.array:
        raise NotImplementedError

class MA2coeff(Model):
    def __init__(self) -> None:
        super().__init__()

    def simulate(self, n = 100) -> np.array:
        """
        uniformly samples n coefficient-pairs from admissable domain by rejection sampling
        """
        i = 0
        theta = np.zeros((2, n))
        while i < n:
            x = np.random.uniform(low = -2, high = 2, size = 1)
            y = np.random.uniform(low = -1, high = 1, size = 1)
            if (x + y > -1) and (x - y < 1):
                theta[0, i], theta[1, i] = x, y 
                i += 1
        return theta

class MA2(Model):
    q = 2
    def __init__(self) -> None:
        super().__init__()

    def simulate(self, parameters: np.array, k = 100, n = 100) -> np.array:
        """
        returns n MA(2) sequences of length k
        coeff: 2 x n array
        z: n x k array
        """
        u = np.random.randn(n, self.q + k)
        z = np.array([u[j,2:] + parameters[0, j]*u[j,1:-1] + parameters[1, j]*u[j,:-2] for j in range(n)])
        return z

class GSDE(SDE.SDE, Model):
    def __init__(self, x0: float, t: int) -> None:
        self.t = t
        super().__init__(x0)

    def set_parameters(self, parameters: np.array) -> None:
        assert parameters.shape[0] == 3, 'parameters should be given as array [[alpha],[lambda1], [lambda2]]'
        alpha, self.lam1, self.lam2 = parameters
        self.alpha = alpha * self.t

    def get_parameters(self) -> tuple:
        return self.alpha, self.lam1, self.lam2

    def implicit_milstein(self, x: np.array, dt: float, dW: float) -> np.array:
        alpha, lam1, lam2 = self.get_parameters()
        Nx = x + alpha * lam1 * lam2 * dt + (2 * alpha * lam2 * x)**(1/2) * dW + .5 * alpha * lam2 * (dW**2 - dt)
        Dx = 1 + alpha * dt
        return Nx / Dx
    
    def simulate(self, parameters: np.array, Nsim = 1) -> np.array:
        results = []
        for i in range(Nsim):
            self.set_parameters(parameters[:, i])
            results.append(self.numerical_solution(M = 10**3, N = 10**4, burn_in = 5 * 10**4)) #NB: vurder valgte verdier
        return np.array(results)

