import numpy as np

class Model:
    def __init__(self) -> None:
        pass

    def simulate(self) -> np.array:
        pass

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

class GammaSDEPrior(Model):
    def __init__(self) -> None:
        super().__init__()
    
    def simulate(self, n = 100) -> np.array:
        alpha = np.random.uniform(0, 0.5, size = n)
        shape = np.random.uniform(1, 3, size = n)
        scale = np.random.uniform(0, 2, size = n)
        return np.array([alpha, shape, scale])

class GammaSDE(Model):
    burn_in = 5 * 10**4
    k = 6 * 24 * 365 * 6 + 2 + burn_in
    def __init__(self, x0) -> None:
        self.x0 = x0
        super().__init__()

    def num_solver(self, input_values):
        M = 10**2
        N = -(- self.k // M)
        t = np.linspace(0, 1, N + 1) #skalert til [0, 1]
        dt = 1 / (N + 1)
        dW = np.sqrt(dt) * np.random.randn(N, M)
        X = np.zeros((N, M))
        X[0,:] = self.x0
        for n in range(N-1):
            X[n + 1,:] = self.implicit_milstein(t[n], X[n,:], dt, dW[n,:], input_values)
        return np.ravel(X, order = 'F')[self.burn_in:self.k] 

    def implicit_milstein(self, t, x, dt, dW, input_values):
        """
        returns one implicit milstein step
        """
        if not all([val > 0 for val in x]):
            raise ValueError('Non-positive value {} at step {}'.format(str(x), str(t * self.k)))
        alpha, shape, scale = input_values
        alpha_hat = alpha * self.k
        theta, epsilon = shape * scale, np.sqrt(2 * alpha_hat * scale)
        Nx = x + alpha_hat * theta * dt + epsilon * x**(1/2) * dW + .25 * epsilon**2 * (dW**2 - dt)
        Dx = 1 + alpha_hat * dt
        return  Nx / Dx
    
    def simulate(self, parameters, n = 100) -> np.array:
        return np.array([self.num_solver(parameters[:,i]) for i in range(n)])