import numpy as np
from matplotlib import pyplot as plt
import Model
import Statistic
import Distance
import Kernel

class Sampler:
    def __init__(self, y: np.array, prior: Model, m: Model, s: Statistic, distance: Distance) -> None:
        self.obs = y
        self.prior = prior
        self.model = m
        self.s = s
        self.distance = distance

    def posterior(self) -> dict:
        raise NotImplementedError

class RejectionSampler(Sampler):
    def __init__(self, y: np.array, prior: Model, m: Model, s: Statistic, distance: Distance) -> None:
        super().__init__(y, prior, m, s, distance)

    def posterior(self, Nsim = 10**6, quant = 0.001, splits = None) -> dict:
        """
        returns approximate posterior + information
        given quantile of n samples are returned
        """
        if splits is not None:
            Nsim = abs( -Nsim // splits)
        else:
            splits = 1

        d = self.prior.get_dim()
        s0 = self.s.statistic(self.obs)
        post = np.zeros((d, 0))
        for i in range(splits):
            proposals = self.prior.simulate(Nsim = Nsim)
            z = self.model.simulate(parameters = proposals, Nsim = Nsim)
            sz = self.s.statistic(z) 
            d = self.distance.dist(sz, s0)
            qval = np.quantile(d, q = quant)
            id = np.where(d < qval)[0]
            post = np.concatenate((post, proposals[:, id]), axis = 1)
        return {'distribution' : post, 'tolerance' : qval}


class MCMCSampler(Sampler):
    def __init__(self, y: np.array, prior: Model, m: Model, s: Statistic, distance: Distance, q: Kernel) -> None:
        self.q = q
        super().__init__(y, prior, m, s, distance)
    
    def posterior(self, tolerance, n = 10**6) -> dict:
        s0 = self.s.statistic(self.obs)
        theta = np.zeros((self.q.get_dim(), n))
        theta[:,0] = self.prior.simulate(n = 1) #OBS: implentere ulike første trekk? evt begynne med rejection sampler
        for i in range(n):
            proposal = self.q.step(theta[:,i])
            z = self.model.simulate(parameters = proposal, n = 1)
            sz = self.s.statistic(z)
            d = self.distance.dist(sz, s0)
            if d < tolerance:
                ratio = self.q.pdf(x = proposal, y = theta[:,i]) / self.q.pdf(x = theta[:,i], y = proposal)
                u = np.random.uniform(self.q.get_dim())
                pass

        return super().posterior()
