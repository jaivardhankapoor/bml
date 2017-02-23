from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

x = np.linspace(-5, 5, 300)
b = 2

def func(X, eta):
	return np.divide(np.exp(np.divide(-np.power(X,2), 2*eta)), np.sqrt(eta*2*np.pi))

def laplace(x, mu, b):
    return np.exp(-np.absolute(x-mu)/b)/(2*b)


y = laplace(x,0,0.5)

plt.plot(x,y)
plt.xlabel('$x$')
plt.ylabel('$p(x|\\gamma=2)$')
plt.savefig('1-2.png')
