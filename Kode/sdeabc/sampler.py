import numpy as np
from matplotlib import pyplot as plt
import sdeabc.Model as Model
import sdeabc.Statistic as Statistic
import sdeabc.Distance as Distance
import time

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
        super().__init__(obs, prior, m, s)

    def get_arrays(self, prior: Model, s: Statistic) -> tuple:
        d1, d2 = prior.get_dim(), s.get_dim()
        return np.zeros((d1, 0)), np.zeros((d2, 0))
    
    def posterior(self, Nsim = 10**6, quant = None, tol = None, splits = None, verbose = False):
        if quant is not None:
            return self.posterior_quant(Nsim = Nsim, quant = quant, splits = splits, verbose = verbose)
        elif tol is not None:
            return self.posterior_tol(Nsim = Nsim, tol = tol, verbose = verbose)
        else:
            raise ValueError


    def posterior_quant(self, Nsim = 10**6, quant = 0.001, splits = None, verbose = False) -> dict:
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

    def posterior_tol(self, Nsim = 10**6, tol = 0.5, verbose = False):
        verboseprint = self.verbosity(verbose)
        s0 = self.s.statistic(self.obs)
        theta, S = self.get_arrays(self.prior, self.s)
        accepted = 0
        for i in range(Nsim):
            verboseprint('{}%'.format((i + 1 )/Nsim * 100))
            proposal = self.prior.simulate(Nsim = 1)
            z = self.model.simulate(parameters = proposal, Nsim = 1, timed = verbose)
            sz = self.s.statistic(z)
            d = self.distance.dist(sz, s0)
            print(d)
            if d <= tol:
                theta = np.concatenate((theta, proposal), axis = 1)
                #S = np.hstack((S, sz))
                accepted += 1
        return {'distribution' : theta, 'statistics': S, 'acceptance_ratio': accepted/Nsim}


class MCMCSampler(Sampler):
    def __init__(self, obs: np.array, prior: Model, q: Model, m: Model, kernel: Model, s: Statistic) -> None:
        self.q, self.kernel = q, kernel
        super().__init__(obs, prior, m, s)
    
    def first_step(self, prior: Model, m: Model, s: Statistic, theta) -> tuple:
        #OBS: implement different first steps?
        if theta is None:
            theta = prior.simulate(Nsim = 1)
        z = m.simulate(parameters = theta, Nsim = 1)
        sz = s.statistic(z)
        return theta[:,0], sz
    
    def get_arrays(self, prior: Model, s: Statistic, n: int) -> tuple:
        d1, d2 = prior.get_dim(), s.get_dim()
        return np.zeros((d1, n)), np.zeros((d2, n))
    
    def posterior(self, n = 10**6, verbose = False, first_step = None) -> dict:
        verboseprint = self.verbosity(verbose)
        prior, q, m, kernel, s = self.prior, self.q, self.model, self.kernel, self.s 
        s0 = s.statistic(self.obs)
        theta, S = self.get_arrays(prior, s, n)

        theta[:, 0], S[:, 0] = self.first_step(prior, m, s, theta = first_step)
        accepted = 0
        print('S0', s0)
        for i in range(n - 1):
            verboseprint('{}%'.format((i + 1 )/n * 100))
            proposal = self.q.simulate(theta[:, i], Nsim = 1)
            z = self.model.simulate(parameters = proposal, Nsim = 1)
            sz = self.s.statistic(z)

            N = prior.logpdf(proposal[:,0]) + kernel.logpdf(x = sz, mu = s0)
            D = prior.logpdf(theta[:, i]) + kernel.logpdf(x = S[:, i], mu = s0)
            print('nåværende theta:', theta[:, i])
            print('forslag', proposal[:,0])
            
            print('nåværende S:', S[:, i])
            print('forslag:', sz)
            print('N - D', N- D)
            a, u = 1, 0
            if N - D <= 0:
                a = np.exp(N - D)
                u = np.random.uniform(low = 0, high = 1, size = 1)
            print('u:', u)
            if u <= a:
                theta[:, i + 1], S[:, i + 1] = proposal[:,0], sz
                accepted += 1
            else:
                theta[:, i + 1], S[:, i + 1] = theta[:, i], S[:, i]
        return {'distribution' : theta, 'statistics': S, 'acceptance_ratio': accepted/n}
    
    def adaptive_posterior(self, covs: np.array, n = 10**6, verbose = False) -> dict:
        results = self.posterior(n = n, verbose = verbose)
        acceptances = []
        for cov in covs:
            self.kernel.set_parameters(cov)
            temp = self.posterior(n = n, verbose = verbose)
            results['distribution'] = np.hstack((results['distribution'], temp['distribution']))
            results['distribution'] = np.hstack((results['distribution'], temp['distribution']))
            acceptances.append(temp['acceptance_ratio'])
        results['acceptance_ratio'] = np.array(acceptances)
        return results
