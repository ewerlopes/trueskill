from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
from trueskill.mathematics import Gaussian


y = +1

def delta(x):
    """Indication function"""
    if x : return 0
    else : return 1

def phi(x):
    return norm.pdf(x,0,1)/norm.cdf(x,0,1)

def lamb(x):
    return phi(x)*(phi(x)+x)

def main():
    domain = np.linspace(-4, 6, 100)
    original_g = norm.pdf(domain, 1.5, 1.5)
    truncated = [delta(y-np.sign(x))*norm.pdf(x, 1.5, 1.5) for x in domain]
    plt.plot(domain, original_g, '.', label="original Gaussian", color='black')
    plt.plot(domain, truncated, '-', label="Truncated", color='black', lw=2)

    mu = 1.5 + y * 1.5 * phi(y*1.5/np.sqrt(1.5))
    var = 1.5 * (1 - lamb(y*1.5/np.sqrt(1.5)))

    approx = [norm.pdf(x, mu, var) for x in domain]
    plt.plot(domain, approx, '-', label="moment matched", color='red', lw=2)
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()

