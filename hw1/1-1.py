from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

x = np.linspace(-8, 8, 300)
eta = np.linspace(1,15,15)

def func(X, eta):
	return np.divide(np.exp(np.divide(-np.power(X,2), 2*eta)), np.sqrt(eta*2*np.pi))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, eta)
zs = np.array([func(x,eta) for x,eta in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z)


ax.set_xlabel('$x$')
ax.set_ylabel('$\\eta$')
ax.set_zlabel('$p(x|\\eta)$')
plt.savefig('1-1.png')

