import numpy as np
import sde
from matplotlib import pyplot as plt
from numerics import brownian


x0, parameters = 4.5, np.array([0.25, 4.38, 1.06])
M = sde.GammaSDE(x0, parameters)

t, W = brownian(M = 10**3, N = 10**5)
X = M.num_solver(t, W)
np.save('pg2', X)

#X = np.load('pg2.npy')
x = np.ravel(X, order ='F')
e = np.cumsum(x) / np.arange(1, len(x) + 1)
plt.plot(e)
plt.show()