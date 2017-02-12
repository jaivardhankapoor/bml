import matplotlib.pyplot as plt
import numpy as np
import sklearn

def gaussian(x, mu, var):
    return np.exp(-np.power(x - mu, 2.) / (2 * var))/np.sqrt(2*np.pi*var)

def laplace(x, mu, b):
    return np.exp(-np.absolute(x-mu)/b)/(2*b)

X = np.linspace(-20,20,300)

y = gaussian(X, 0, 100)/2 + gaussian(X,0,1)/2

a = 'p(w|\\sigma_{slab}=100,\\sigma_{spike}=1)'

plt.xlabel('$w$')
plt.ylabel('$%s$'%a)


plt.plot(X,y)
plt.savefig('4-2.png')

