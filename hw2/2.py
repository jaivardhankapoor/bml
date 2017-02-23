import numpy as np
from scipy.special import gamma
from matplotlib import pyplot as plt
from matplotlib import legend
from math import e, pi, sqrt, pow

def linspace(start, stop, n):
    if n == 1:
        yield stop
        return
    h = (stop - start) / (n - 1)
    for i in range(n):
        yield start + h * i


def gamma_approx(a):
    return [sqrt(2*pi*e)*pow((i-1.)/e, i-1./2.) for i in a]

x = np.linspace(1,5,100)
plt.plot(x,gamma_approx(x), '-r', label='$y = \\Gamma_{approx}(x)$')
plt.plot(x,gamma(x), '-b', label='$y = \\Gamma(x)$')
# print(gamma_approx([7.,8.,9.]))
plt.legend(loc='best')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('2(a).png')

plt.clf()
plt.cla()
plt.close()

x = np.linspace(1,8,100)
plt.plot(x,np.log(gamma_approx(x)), '-r', label='$y = ln(\\Gamma_{approx}(x))$')
plt.plot(x,np.log(gamma(x)), '-b', label='$y = ln(\\Gamma(x))$')
plt.legend(loc='best')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('2(b).png')
