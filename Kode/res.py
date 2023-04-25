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

n = y.shape[1]
e = np.array([np.mean(y[:,:i]) for i in range(1, n)])
v = np.array([np.var(y[:,:i]) for i in range(1, n)])
t = np.arange(0, n - 1)

m_fig, m_ax = plt.subplots(figsize = (8, 7))
m_ax.plot(t, e)
m_ax.grid()
m_ax.set_xlabel('N')
m_ax.set_ylabel(r'Mean($X_{1:N}$)')
m_fig.savefig('Figurer/gsde_mean_values.png')

m_fig, m_ax = plt.subplots(figsize = (8, 7))
m_ax.plot(t, v)
m_ax.grid()
m_ax.set_xlabel('N')
m_ax.set_ylabel(r'Var($X_{1:N}$)')
m_fig.savefig('Figurer/gsde_var_values.png')

"""scheme = sampler.RejectionSampler(y = y, prior = prior, m = m, s = stat, distance = d)
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
plt.savefig('SDE_RS_test2004.png')"""