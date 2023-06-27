import numpy as np
import sdeabc.SDE as SDE
from scipy.stats import gamma, uniform
from scipy.stats import multivariate_normal as mvnr
from sdeabc.Statistic import ac
import time

class Model:
    def __init__(self) -> None:
        self.dim = None
        pass

    def get_dim(self) -> int:
        return self.dim

    def simulate(self, Nsim) -> np.array:
        raise NotImplementedError

    def logpdf(self, x) -> np.array:
        raise NotImplementedError
    
    def timeit(self, timed: bool):
        if timed:
            starttime = time.time()
            def get_time():
                print('Model run took {} s'.format(time.time() - starttime))
        else:
            get_time = lambda: None
        return get_time


class JointModel(Model):
    def __init__(self, models: list) -> None:
        self.models = models
        self.dim = sum([m.get_dim() for m in models])
        self.size = len(models)
    
    def get_dim(self) -> int:
        return self.dim
    
    def get_models(self):
        return self.models

    def get_size(self):
        return self.size

    def simulate(self, Nsim) -> np.array:
        """
        NB: mangler funksjonalitet for modeller som tar inn parametere i simulate + kan hende det kun går for Nsim = 1
        """
        d = self.get_dim()
        models = self.get_models()
        simulations = np.zeros((0))
        for m in models:
            simulations = np.append(simulations, m.simulate(Nsim))
        return np.array(simulations).reshape(d, -1)

    def logpdf(self, x) -> np.array:
        """
        logpdf : summen av logpdf-ene
        """
        size, models = self.get_size(), self.get_models()
        lpdf = 0
        dims = [0]
        for i in range(size):
            m = models[i]
            dims.append(m.get_dim())
            lpdf += m.logpdf(x[dims[i]:(dims[i] + dims[i + 1])])
        return lpdf


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
    def __init__(self, x0: float, t: int, M: int, N: int, burn_in = 0, with_stats = False, correlated_solutions = False) -> None:
        self.t, self.M, self.N = t, M, N
        self.burn_in = burn_in
        super().__init__(x0 = x0, t = t, with_stats = with_stats, correlated_solutions = correlated_solutions)

    def set_parameters(self, parameters: np.array) -> None:
        """
        parameters: 3 x 1 np.array
        """
        assert parameters.shape[0] == 3, 'parameters should be given as array [[alpha],[lambda1], [lambda2]]'
        a, self.lam1, self.lam2 = parameters
        self.alpha = a  #* self.t

    def get_parameters(self) -> tuple:
        return self.alpha, self.lam1, self.lam2

    def implicit_milstein(self, x: np.array, dt: float, dW: float) -> np.array:
        alpha, lam1, lam2 = self.get_parameters()
        Nx = x + alpha * lam1 * lam2 * dt + (2 * alpha * lam2 * x)**(1/2) * dW + .5 * alpha * lam2 * (dW**2 - dt)
        Dx = 1 + alpha * dt
        return Nx / Dx
    
    def statistics(self, x: np.array):
       m = np.mean(x)
       sd = np.std(x)
       c = np.mean([ac(i, lag = 1) for i in x])
       return np.reshape(np.array([c, m, sd]), (3))
    
    def simulate(self, parameters: np.array, Nsim = 1, timed = False) -> np.array:
        """
        parameters: 3 x Nsim np.array
        results: Nsim x k array
        """
        M, N = self.M, self.N
        burn_in = self.burn_in
        get_time = self.timeit(timed)
        results = []
        for i in range(Nsim):
            self.set_parameters(parameters[:, i])
            results.append(self.numerical_solution(M = M, N = N, burn_in =  burn_in))
        get_time()
        if Nsim == 1:
            return results[0]
        else:
            return np.array(results)


class Gammadist(Model):
    def __init__(self, parameters: np.array) -> None:
        """
        parameters: 2 x d parameters
        """
        assert parameters.shape[0] == 2, 'parameters should be given as [[shape], [scale]]'
        self.dim = parameters.shape[1]
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
        NB only for use in mcmc ratio rn
        x : (d, ) array
        """
        """if x[0] <= 1:
            return - np.infty"""
        a, b = self.get_parameters()
        return np.sum((a - 1) * np.log(x) - (x / b))

class UniformDist(Model):
    def __init__(self, parameters: np.array) -> None:
        """
        parameters: 2 x d parameters
        """
        assert parameters.shape[0] ==  2, 'parameters should be given as [[low], [high]]'
        self.dim = parameters.shape[1]
        self.low, self.high = parameters

    def get_dim(self) -> int:
        return self.dim

    def get_parameters(self) -> tuple:
        return self.low, self.high
    
    def simulate(self, Nsim) -> np.array:
        d = self.get_dim()
        low, high = self.get_parameters()
        rv = uniform.rvs(loc = low, scale = high, size = (Nsim, d)).ravel(order = 'F')
        return rv.reshape(d, -1)

    def logpdf(self, x) -> np.array:
        low, high = self.get_parameters()
        lp = uniform.logpdf(x, loc = low, scale = high)
        return lp

class RandomWalk(Model):
    def __init__(self, covariance: np.array) -> None:
        self.dim = covariance.shape[0]
        self.cov = np.diag(covariance)
    
    def set_parameters(self, covariance: np.array) -> None:
        assert covariance.shape[0] == self.get_dim(), 'new covariance matrix must have same dimensions as previous'
        self.cov = covariance

    def set_parameters(self, covariance):
        assert self.get_dim() == covariance.shape[0], 'new covariance structure must have same dimensions as previous'
        self.cov = covariance
        
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
        """
        x: d x 
        mu: d x 
        """
        if (any(x) <= 0) or (x[0] >= 1):
            return 0
        cov = self.get_parameters()
        return mvnr.logpdf(x = x, mean = mu, cov = cov)


