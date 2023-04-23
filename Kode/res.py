import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import Model
import Distance
import Statistic
import sampler

prior = Model.GammaSDEPrior()
m = Model.GammaSDE(x0 = 4)
stat = Statistic.StationaryStats()
d = Distance.EuclidianDistance()

theta = np.array([[0.25], [2.1], [1.4]])
y = m.simulate(parameters = theta, n = 1)

scheme = sampler.RejectionSampler(y = y, prior = prior, m = m, s = stat, distance = d)
result = scheme.posterior(n = 10**5, quant = 0.001)
par = result['distribution']
print(result['tolerance'])

np.save('SDE_RS_test2004', par)


fig, ax = plt.subplots(nrows = 3, figsize = (7, 21))
ax[0].hist(
    par[0], 
    bins = 25,
    density = True)
ax[1].hist(
    par[1], 
    bins = 25,
    density = True)
ax[2].hist(
    par[2], 
    bins = 25,
    density = True)
plt.savefig('SDE_RS_test2004.png')