import numpy as np
from proposal import RandomWalk
import sde
import stats

def likelihood_free_mh(x, tol, N = 100):
    Sx = stats.stationary_statistics(x)
    q = RandomWalk()
    x0 = x[0]
    prior = None
    theta0 = prior
    theta = np.zeros((N, len(theta0)))
    M = sde.GammaSDE(x0 = x0, parameters = theta)

    for n in N:
        theta[n + 1] = theta[n]
        pd = prior(theta)
        prop = q.step(theta)
        y = M.simulate(parameters = prop)

        Sy = stats.stationary_statistics(y)
        distance = np.linalg.norm((Sx - Sy))
        if distance <= tol:
            ratio = prior(prop) * q.density(theta, prop) / (pd * q.density(prop, theta))
            u = np.random.uniform(0, 1, 1)
            if np.min(1, ratio) <= u:
                theta[n + 1] = prop
        
    return

