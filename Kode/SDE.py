import numpy as np
from numerics import brownian

class SDE:
    def __init__(self, x0, **parameters) -> None:
        self.x0 = x0
        
    def drift(self, t, x) -> float: 
        raise NotImplementedError

    def diffusion(self, t, x) -> float:
        raise NotImplementedError

    def update_parameters(self, **parameters) -> None:
        raise NotImplementedError
    
    def get_parameters(self):
        raise NotImplementedError
    
    def euler_maruyama(self, t, x, dt, dW):
        """
        returns one euler-maruyama step
        """
        return x + self.drift(t, x) * dt + self.diffusion(t, x) * dW 

    def implicit_milstein(self, t, x, dt, dW):
        raise NotImplementedError
    
    def num_solver(self, t, W):
        """
        returns numerical solution of SDE Y on [0, t_end]
        """
        N, M = np.shape(W)
        X = np.zeros((N, M))
        dt = t[1]-t[0]
        
        X[0,:] = self.x0
        for n in range(N-1):
            dW = W[n + 1,:] - W[n,:]
            X[n + 1,:] = self.implicit_milstein(t[n], X[n,:], dt, dW)
        return X
    
    def simulate(self, parameters = None):
        t, W = brownian(t_end = 1, N = 10**8)
        if parameters is not None:
            self.update_parameters(parameters)
        X = self.num_solver(t, W)
        return X

class MeanRevertingSDE(SDE):
    def __init__(self, x0, kappa, theta, epsilon) -> None:
        self.kappa, self.theta, self.epsilon = kappa, theta, epsilon
        super().__init__(x0)
        return
    
    def get_parameters(self):
        return self.kappa, self.theta, self.epsilon
    
    def update_parameters(self, kappa, theta, epsilon) -> None:
        self.kappa, self.theta, self.epsilon = kappa, theta, epsilon
        return

    def drift(self, t, x) -> float:
        kappa, theta, epsilon = self.get_parameters()
        return kappa * (theta - x)
    
    def diffusion(self, t, x) -> float:
        kappa, theta, epsilon = self.get_parameters()
        return epsilon * np.sqrt(x)

    def implicit_milstein(self, t, x, dt, dW):
        kappa, theta, epsilon = self.get_parameters()
        N = x + kappa * theta * dt + epsilon * x**(1/2) * dW + .25 * epsilon**2 * (dW**2 - dt)
        D = 1 + kappa * dt
        return  N / D

class GammaSDE(MeanRevertingSDE):
    def __init__(self, x0, parameters) -> None:
        assert parameters.shape == (3,), 'GammaSDE requires 3 parameters'
        alpha, a, b = parameters
        super().__init__(x0 = x0, kappa = alpha, theta = a * b, epsilon = (2 * alpha * b)**(1/2))
    
    def update_parameters(self, parameters) -> None:
        assert parameters.shape == (3,), 'GammaSDE requires 3 parameters'
        alpha, a, b = parameters
        return super().update_parameters(kappa = alpha, theta = a * b, epsilon = (2 * alpha * b)**(1/2))
