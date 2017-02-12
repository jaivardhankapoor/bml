import matplotlib.pyplot as plt
import numpy as np
import sklearn

def gaussian(x, mu, var):
    return np.exp(-np.power(x - mu, 2.) / (2 * var))/np.sqrt(2*np.pi*var)

def laplace(x, mu, b):
    return np.exp(-np.absolute(x-mu)/b)/(2*b)

w = np.linspace(0,6,300)

y = np.divide(gaussian(3, w, 0.01)*(gaussian(w,0, 100) + gaussian(w,0, 1)), gaussian(3, 0, 0.01+100)+gaussian(3,0,0.01+1))

a = 'p(w|x=3,\\sigma_{slab}=100,\\sigma_{spike}=1,\\rho^2=0.01)'

plt.xlabel('$w$')
plt.ylabel('$%s$'%a)


plt.plot(w,y)
plt.savefig('4-5.png')
