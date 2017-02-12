from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

x = np.linspace(-8, 8, 300)
eta = 3

def func(X, eta):
	return np.divide(np.exp(np.divide(-np.power(X,2), 2*eta)), np.sqrt(eta*2*np.pi))


#ax = fig.add_subplot(111, projection='3d')

#X, Y = np.meshgrid(x, eta)
#zs = np.array([func(x,eta) for x,eta in zip(np.ravel(X), np.ravel(Y))])
#Z = zs.reshape(X.shape)
#ax.plot_surface(X, Y, Z)

y = func(x,eta)
plt.plot(x,y)
plt.xlabel('$x$')
#ax.set_ylabel('$\\eta$')
plt.ylabel('$p(x|\\eta=3)$')
plt.savefig('1-1.png')

