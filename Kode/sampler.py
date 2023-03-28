import numpy as np
from proposal import RandomWalk, Prior_gammapar
import sde
import stats
from matplotlib import pyplot as plt

def likelihood_free_mh(x, tol = 15, N = 100):
    Sx = stats.stationary_statistics(x)
    q = RandomWalk(sigma = 0.1)
    x0 = x[0]
    prior = Prior_gammapar()
    theta0 = prior.step()
    D = len(theta0)
    theta = np.zeros((N, D))
    M = sde.GammaSDE(x0 = x0, parameters = theta0)
    distance = np.inf
    while distance > tol:
        theta0 = prior.step()
        y = M.simulate(parameters = theta0)
        Sy = stats.stationary_statistics(y)
        distance = np.linalg.norm((Sx - Sy))
    theta[0] = theta0
    for n in range(N-1):
        theta[n + 1] = theta[n]
        pd = prior.density(theta[n])
        prop = q.step(theta[n])
        y = M.simulate(parameters = prop)

        Sy = stats.stationary_statistics(y)
        distance = np.linalg.norm((Sx - Sy))
        print('distance:', distance)
        if distance <= tol:
            ratio = prior.density(prop) * q.density(theta[n], prop) / (pd * q.density(prop, theta[n]))

            u = np.random.uniform(0, 1, size = D)
            comp = np.minimum(np.ones(D), ratio)
            for i in range(D):
                if u[i] <= comp[i]:
                    theta[n + 1, i] = prop[i]
        
    return theta

x = np.load('Kode\pg2.npy')
par = likelihood_free_mh(x)

print(par)

print(np.mean(par, axis = 0))

fig, ax = plt.subplots(ncols = 3, figsize = (18, 6))
ax[0].hist(par[:,0])
ax[1].hist(par[:,1])
ax[2].hist(par[:,2])
plt.show()