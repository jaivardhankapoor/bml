import matplotlib.pyplot as plt
import numpy as np
import sklearn

def gaussian(x, mu, var):
    return np.exp(-np.power(x - mu, 2.) / (2 * var))/np.sqrt(2*np.pi*var)

def laplace(x, mu, b):
    return np.exp(-np.absolute(x-mu)/b)/(2*b)

x = np.linspace(-6,6,300)

y = np.divide(gaussian(x, 0, 0.01+100), gaussian(x, 0, 0.01+100)+gaussian(x,0,0.01+1))

a = 'p(b=1|x,\\sigma_{slab}=100,\\sigma_{spike}=1,\\rho^2=0.01)'

plt.xlabel('$x$')
plt.ylabel('$%s$'%a)


plt.plot(x,y)
plt.savefig('4-3.png')
