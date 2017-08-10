from scipy.special import iv
import numpy as np

def t1():
    class A:

        def __init__(self):
            self.mu = 1
            self.bag = {}

        def add(self, b):
            self['i'] = b

        def p(self):
            return self

        def __getitem__(self, factor):
            return self.bag[factor]

        def __setitem__(self, factor, message):
            self.bag[factor] = message


    a = A()
    a.add(20)
    print a['i']
    print a.p()





def computePoissonWinProbLowerBound(lambdaI, lambdaJ):
    pPoisson = 0

    for tempCounter in range(1,75):# approximation to calculate the winProb. Lower bound
        pPoisson = pPoisson + np.exp(-(lambdaI+lambdaJ)) * \
                              (lambdaI/lambdaJ)**(tempCounter/2) *\
                              iv(tempCounter, 2*np.sqrt(lambdaI*lambdaJ))
    return pPoisson



from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
from trueskill.mathematics import Gaussian


def version1Efffort():
    diff = Gaussian(mu=3, sigma=2)
    epsilon1 = Gaussian(mu=0, sigma=1)
    epsilon2 = Gaussian(mu=8, sigma=1)

    sub_epsilon = Gaussian(mu=epsilon1.mu - epsilon2.mu, sigma=np.sqrt(epsilon1.sigma**2 + epsilon2.sigma**2))

    sub = Gaussian(mu=diff.mu - sub_epsilon.mu, sigma=np.sqrt(diff.sigma**2 + sub_epsilon.sigma**2))

    domain = np.linspace(-10, 10, 200)

    functions = [epsilon1, epsilon2, sub_epsilon]

    plt.figure(figsize=(20,6))

    plt.subplot(141)
    for i in range(len(functions)):
        y = [norm.pdf(x, functions[i].mu, functions[i].sigma) for x in domain]
        if i < 2:
            plt.plot(domain,y, ".-.", label="epsilon_{}".format(i+1))
        else:
            plt.plot(domain, y, label="subtraction")
    plt.legend(loc="best", fontsize=9)

    #########################
    plt.subplot(142)
    functions = [diff, sub]

    for i in range(len(functions)):
        y = [norm.pdf(x, functions[i].mu, functions[i].sigma) for x in domain]
        if i == 0:
            plt.plot(domain, y, label="diff")
        else:
            plt.plot(domain, y, label="sub")
    plt.legend(loc="best", fontsize=9)

    ##########################
    plt.subplot(143)
    y = [norm.cdf(x, diff.mu, diff.sigma) for x in domain]
    plt.plot(domain, y, label="diff cdf")
    plt.legend(loc="best", fontsize=9)

    ##########################
    plt.subplot(144)
    y = [norm.cdf(x, sub.mu, sub.sigma) for x in domain]
    plt.plot(domain, y, label="summ cdf")

    plt.legend(loc="best", fontsize=9)
    plt.show()


def version2Effort():

    perf1    = Gaussian(mu=10, sigma=2)
    epsilon1 = Gaussian(mu=0, sigma=1)

    G1 =  Gaussian(mu=perf1.mu + epsilon1.mu, sigma=np.sqrt(perf1.sigma ** 2 + epsilon1.sigma ** 2))

    perf2    = Gaussian(mu=4, sigma=2)
    epsilon2 = Gaussian(mu=0, sigma=1)

    G2 = Gaussian(mu=perf2.mu + epsilon2.mu, sigma=np.sqrt(perf2.sigma ** 2 + epsilon2.sigma ** 2))

    sub_G = Gaussian(mu=G1.mu - G2.mu, sigma=np.sqrt(G1.sigma ** 2 + G2.sigma ** 2))

    domain = np.linspace(-20, 20, 400)

    functions = [G1, G2]

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    for i in range(len(functions)):
        y = [norm.pdf(x, functions[i].mu, functions[i].sigma) for x in domain]
        plt.plot(domain, y, ".-.", label="G_{}".format(i + 1))
    plt.legend(loc="best", fontsize=9)
    #########################

    plt.subplot(132)

    y = [norm.pdf(x, sub_G.mu, sub_G.sigma) for x in domain]
    plt.plot(domain, y, label="sub")
    plt.legend(loc="best", fontsize=9)

    ##########################

    plt.subplot(133)

    y = [norm.cdf(x, sub_G.mu, sub_G.sigma) for x in domain]
    plt.plot(domain, y, label="cdf")
    plt.legend(loc="best", fontsize=9)
    ##########################

    plt.show()

version1Efffort()
