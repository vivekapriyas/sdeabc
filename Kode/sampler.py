import numpy as np
from matplotlib import pyplot as plt
import Model
import Statistic
import Distance
import Kernel

class Sampler:
    def __init__(self, obs: np.array, prior: Model, m: Model, s: Statistic) -> None:
        self.obs = obs
        self.prior = prior
        self.model = m
        self.s = s

    def posterior(self) -> dict:
        raise NotImplementedError
    
    def verbosity(self, verbose: bool):
        if verbose:
            def verboseprint(*args):
                for arg in args:
                    print(arg)
                return
        else:
            verboseprint = lambda *args: None
        return verboseprint


class RejectionSampler(Sampler):
    def __init__(self, obs: np.array, prior: Model, m: Model, s: Statistic, distance: Distance) -> None:
        self.distance = distance
        super().__init__(obs, prior, m, s, distance)

    def posterior(self, Nsim = 10**6, quant = 0.001, splits = None, verbose = False) -> dict:
        """
        returns approximate posterior + information
        given quantile of n samples are returned
        """
        verboseprint = self.verbosity(verbose)
        if splits is not None:
            Nsim = abs( -Nsim // splits)
        else:
            splits = 1

        d = self.prior.get_dim()
        s0 = self.s.statistic(self.obs)
        post = np.zeros((d, 0))
        for i in range(splits):
            verboseprint('Starting split {} out of {}'.format(i + 1, splits))
            proposals = self.prior.simulate(Nsim = Nsim)
            z = self.model.simulate(parameters = proposals, Nsim = Nsim)
            sz = self.s.statistic(z) 
            d = self.distance.dist(sz, s0)
            qval = np.quantile(d, q = quant)
            id = np.where(d < qval)[0]
            post = np.concatenate((post, proposals[:, id]), axis = 1)
            verboseprint('In split {} the tolerance was {}.'.format(i + 1, qval))
        return {'distribution' : post, 'tolerance' : qval}


class MCMCSampler(Sampler):
    def __init__(self, obs: np.array, prior: Model, q: Model, m: Model, kernel: Model, s: Statistic) -> None:
        self.q, self.kernel = q, kernel
        super().__init__(obs, prior, m, s)
    
    def first_step(self, prior: Model, m: Model, s: Statistic) -> tuple:
        #OBS: implement different first steps?
        theta = prior.simulate(Nsim = 1)
        z = m.simulate(parameters = theta, Nsim = 1)
        sz = s.statistic(z)
        return theta[:,0], sz

    def get_arrays(self, prior: Model, s: Statistic, n: int) -> tuple:
        d1, d2 = prior.get_dim(), s.get_dim()
        return np.zeros((d1, n)), np.zeros((d2, n))

    def posterior(self, n = 10**6, verbose = False) -> dict:
        verboseprint = self.verbosity(verbose)
        prior, q, m, kernel, s = self.prior, self.q, self.model, self.kernel, self.s 
        s0 = s.statistic(self.obs)
        theta, S = self.get_arrays(prior, s, n)
        theta[:, 0], S[:, 0] = self.first_step(prior, m, s)

        for i in range(n - 1):
            verboseprint('{}%'.format((i + 1 )/n * 100))
            proposal = self.q.simulate(theta[:, i], Nsim = 1)
            z = self.model.simulate(parameters = proposal, Nsim = 1)
            sz = self.s.statistic(z)

            N = prior.logpdf(proposal) + kernel.logpdf(x = sz, mu = s0)
            D = prior.logpdf(theta[:, i]) + kernel.logpdf(x = S[:, i], mu = s0)
            
            a, u = 1, 0
            if N - D <= 0:
                a = np.exp(N - D)
                u = np.random.uniform(1)
            if u <= a:
                theta[:, i + 1], S[:, i + 1] = proposal[:,0], sz
            else:
                theta[:, i + 1], S[:, i + 1] = theta[:, i], S[:, i]

        return {'distribution' : theta, 'statistics': S}

