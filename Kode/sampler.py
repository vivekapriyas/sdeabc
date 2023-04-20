import numpy as np
from matplotlib import pyplot as plt
import Model
import Statistic
import Distance

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

    def posterior(self, n = 10**6, quant = 0.001) -> dict:
        """
        returns approximate posterior + information
        given quantile of n samples are returned
        """
        s0 = self.s.statistic(self.obs)
        proposals = self.prior.simulate(n = n)
        z = self.model.simulate(parameters = proposals, n = n)
        sz = self.s.statistic(z) 
        print(sz)
        d = self.distance.dist(sz, s0)
        qval = np.quantile(d, q = quant)
        i = np.where(d < qval)[0]
        par = proposals[:, i]
        return {'distribution' : par, 'tolerance' : qval}
    