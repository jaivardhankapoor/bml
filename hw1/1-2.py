from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

x = np.linspace(-8, 8, 300)
b = np.linspace(2,0.5,150)

def func(X, eta):
	return np.divide(np.exp(np.divide(-np.power(X,2), 2*eta)), np.sqrt(eta*2*np.pi))

def laplace(x, mu, b):
    return np.exp(-np.absolute(x-mu)/b)/(2*b)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, b)
zs = np.array([laplace(x,0,1/b) for x,b in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, np.divide(1,Y), Z)


ax.set_xlabel('$x$')
ax.set_ylabel('$\\gamma$')
ax.set_zlabel('$p(x|\\gamma)$')
plt.savefig('1-2.png')

