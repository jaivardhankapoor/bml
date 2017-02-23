import numpy as np
from matplotlib import pyplot as plt

x = np.array([-2.23, -1.30, -0.42, 0.30, 0.33, 0.52, 0.87, 1.80, 2.74, 3.62]).T
y = np.array([1.01, 0.69, -0.66, -1.34, -1.75, -0.98, 0.25, 1.57, 1.65, 1.51]).T
n, beta = 10, 4.


def phi(x, k):
    return np.vstack([np.power(x,i) for i in range(k+1)]).T

def normal(x, mu, cov):
    D = np.shape(cov)[0]
    return 1./(np.sqrt(np.power(2*np.pi, D)*np.linalg.det(cov)))*np.exp(-0.5*(x - mu).dot(np.linalg.inv(cov)).dot(x.T-mu.T))

#########################PART 1###############################

def plot_post(k):
    lambda_ = 1
    X = phi(x,k)
    w = np.zeros(np.shape(X)[1])
    w_mu = np.linalg.inv((X.T).dot(X) + lambda_*np.eye(k+1)/beta).dot((X.T)).dot(y)
    w_cov = np.linalg.inv(beta*(X.T).dot(X) + lambda_*np.eye(k+1))

    plt.scatter(x,y)

    X_test = phi(np.linspace(-4, 4, 200), k)


    y_test = pred_post_mean = np.array([X_test[i,:].dot(w_mu) for i in range(200)])

    pred_post_var = 1./beta + np.array([X_test[i,:].dot(w_cov.dot(X_test[i,:].T)) for i in range(200)])

    # print(np.shape(pred_post_var))

    error = 2*pred_post_var

    plt.plot(np.linspace(-4,4,200), y_test)
    plt.fill_between(np.linspace(-4,4,200), y_test+error, y_test-error, facecolor='grey', alpha=0.4)
    plt.title('Predictive posteror with error margin of $2\\sigma$ for $k = ${}'.format(k))
    plt.xlabel('$x_{*}$')
    plt.ylabel('$y_{*}$')
    plt.savefig('5-1-{}.png'.format(k))

    plt.clf()
    plt.cla()
    plt.close()

for i in range(1,4):
    plot_post(i)
########################PART 2###############################
def marginal(k):
    lambda_ = 1
    X = phi(x,k)
    w = np.zeros(np.shape(X)[1])
    w_mu = np.linalg.inv((X.T).dot(X) + lambda_*np.eye(k+1)/beta).dot((X.T)).dot(y)
    w_cov = np.linalg.inv(beta*(X.T).dot(X) + lambda_*np.eye(k+1))
    mar_lik = normal(y, w_mu[0]*np.ones(n), beta*np.eye(n) + X[:,2:].dot(X[:,2:].T))
    return mar_lik

for i in range(1,4):
    print('marginal likelihood for k = {} is {}'.format(i, marginal(i)))
