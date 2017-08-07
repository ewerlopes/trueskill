from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt


y = +1
u = float("inf")
epsilon = 0


def indicator(x,l,u):
    if (x > l and x < u):
        return 1
    else: return 0


def rectGaussian(x,mu,var,l,u):
    return indicator(x,l,u) * \
            (norm.pdf(x,mu,np.sqrt(var)) / \
             (norm.cdf(u,mu,np.sqrt(var))- norm.cdf(l,mu,np.sqrt(var))))


def delta(x):
    """Indication function"""
    if x : return 0
    else : return 1

def phi(x):
    return norm.pdf(x)/norm.cdf(x)

def lamb(x):
    return phi(x)*(phi(x)+x)

def main():

    o_mu  = 1.3
    o_var = 1.7

    domain = np.linspace(-4, 6, 1000)

    original_g = [norm.pdf(x, o_mu, np.sqrt(o_var)) for x in domain]
    truncated  = [delta(y-np.sign(x)) * norm.pdf(x, o_mu, np.sqrt(o_var)) for x in domain]
    rectified  = [rectGaussian(x, o_mu, o_var, epsilon, u) for x in domain]

    plt.figure(figsize=(6,4))
    plt.plot(domain, original_g, '-', label="original Gaussian", color='black', lw=0.5)
    plt.plot(domain, truncated, '-', label="Truncated", color='black', lw=1)
    plt.plot(domain, rectified, '-', label="rectified", color='blue', lw=1)

    mu = o_mu + y * np.sqrt(o_var) * phi((y*o_mu)/np.sqrt(o_var))
    var = o_var * (1 - lamb((y*o_mu)/np.sqrt(o_var)))

    approx = [norm.pdf(x, mu, var) for x in domain]
    plt.plot(domain, approx, '-', label="moment matched", color='red', lw=2)
    plt.legend(loc='best', fontsize=8)
    plt.xlim((-3,4.5))
    plt.show()

if __name__ == '__main__':
    main()

