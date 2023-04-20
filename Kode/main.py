import numpy as np
from matplotlib import pyplot as plt
from sde import GammaSDE
from statsmodels.tsa.stattools import acf

x0, alpha, shape, scale = 4.0, 0.25, 2.1, 1.4
T = 6 * 24 * 365 #* 6 + 2
parameters = np.array([alpha * T, shape, scale])

M = GammaSDE(x0, parameters)
Y = M.simulate(M = 2, N = int(T/2), burn_in = 0)
t = np.arange(0, T + 2)
m = np.cumsum(Y) / np.arange(1, T + 3)
sd = np.array([np.std(Y[:i]) for i in range(1, T + 3)])
c = acf(Y, nlags = 4)
print(c)

fig, ax = plt.subplots(nrows = 2)
ax[0].plot(t, m)
ax[1].plot(t, sd)
plt.show()
