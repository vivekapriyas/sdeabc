import numpy as np
import SDE
from scipy.stats import gamma, multivariate_normal as mvnr
from matplotlib import pyplot as plt

class Model:
    def __init__(self) -> None:
        pass

    def get_dim(self) -> int:
        raise NotImplementedError

    def simulate(self, Nsim) -> np.array:
        raise NotImplementedError

    def logpdf(self, x) -> np.array:
        raise NotImplementedError

class MA2coeff(Model):
    def __init__(self) -> None:
        super().__init__()

    def get_dim(self):
        return 2

    def simulate(self, Nsim = 100) -> np.array:
        """
        uniformly samples n coefficient-pairs from admissable domain by rejection sampling
        """
        i = 0
        theta = np.zeros((2, Nsim))
        while i < Nsim:
            x = np.random.uniform(low = -2, high = 2, size = 1)
            y = np.random.uniform(low = -1, high = 1, size = 1)
            if (x + y > -1) and (x - y < 1):
                theta[0, i], theta[1, i] = x, y 
                i += 1
        return theta

class MA2(Model):
    q = 2
    def __init__(self, k = 100) -> None:
        self.k = k
        super().__init__()

    def get_dim(self) -> int:
        return self.k

    def simulate(self, parameters: np.array, Nsim = 100) -> np.array:
        """
        returns n MA(2) sequences of length k
        coeff: 2 x n array
        z: n x k array
        """
        k = self.get_dim()
        u = np.random.randn(Nsim, self.q + k)
        z = np.array([u[j,2:] + parameters[0, j]*u[j,1:-1] + parameters[1, j]*u[j,:-2] for j in range(Nsim)])
        return z

class GSDE(SDE.SDE, Model):
    def __init__(self, x0: float, t: int) -> None:
        self.t = t
        super().__init__(x0)

    def set_parameters(self, parameters: np.array) -> None:
        """
        parameters: 3 x 1 np.array
        """
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
        """
        parameters: 3 x Nsim np.array
        results: Nsim x k array
        """
        results = []
        for i in range(Nsim):
            self.set_parameters(parameters[:, i])
            results.append(self.numerical_solution(M = 10**3, N = 10**4, burn_in =  5 * 10**4)) #NB: vurder valgte verdier
        return np.array(results)

class Gammadist(Model):
    def __init__(self, parameters: np.array) -> None:
        """
        parameters: 2 x d parameters
        """
        assert parameters.shape[0] == 2, 'parameters should be given as [[shape], [scale]]'
        self.dim = parameters.shape[2]
        self.alpha, self.beta = parameters
    
    def get_dim(self) -> int:
        return self.dim
    
    def get_parameters(self) -> tuple:
        """
        alpha: (d, ) array
        beta: (d, ) array
        """
        return self.alpha, self.beta
    
    def simulate(self, Nsim) -> np.array:
        """
        returns 2 x Nsim np.array
        """
        d = self.get_dim()
        alpha, beta = self.get_parameters()
        rv = gamma.rvs(a = alpha, scale =  beta, size = (Nsim, d)).ravel(order = 'F')
        return rv.reshape(d, -1)
    
    def logpdf(self, x) -> np.array:
        """
        x : (d, ) array
        """
        d = self.get_dim()
        assert x.shape[0] == d, 'pdf input must have shape[0] == {}'.format(d)
        alpha, beta = self.get_parameters()
        return np.sum(gamma.logpdf(x = x, a = alpha, scale = beta))

class RandomWalk(Model):
    def __init__(self, covariance: np.array) -> None:
        self.dim = covariance.shape[0]
        self.cov = np.diag(covariance)
    
    def get_dim(self) -> int:
        return self.dim
    
    def get_parameters(self) -> np.array:
        return self.cov
    
    def simulate(self, mu, Nsim) -> np.array:
        """
        returns d x Nsim np.array
        """
        d = self.get_dim()
        cov = self.get_parameters()
        rv = mvnr.rvs(mean = mu, cov = cov, size = Nsim).ravel(order = 'F')
        return  rv.reshape(d, -1)
    
    def logpdf(self, x, mu) -> np.array:
        cov = self.get_parameters()
        return mvnr.logpdf(x = x, mean = mu, cov = cov)
    