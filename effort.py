from scipy.stats import norm, beta
from matplotlib import pyplot as plt
import numpy as np

mu1 = 3
mu2 = 5
sigma = 1

domain = np.linspace(-10, 10, 200)
y = [norm.pdf(x) for x in domain]

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(domain, [norm.pdf(x, mu1, sigma) for x in domain], label="$P1$")
plt.plot(domain, [norm.pdf(x, mu2, sigma) for x in domain], label="$P2$")
plt.plot(domain, [norm.pdf(x, mu1-mu2, np.sqrt(sigma**2+sigma**2)) for x in domain], label="$\mu1 - \mu2$")
plt.legend(loc="best", fontsize=8)

plt.subplot(132)
z = [x / np.sqrt(sigma ** 2 + sigma ** 2) for x in domain]
plt.plot(z, [norm.pdf(x, mu1 - mu2, 1) for x in z], label="$N(x; w1-w2, 1)$")
plt.plot(domain, y, label="$N(x;0,1)$")
plt.plot(domain, [norm.cdf(x) for x in domain], label="$\Phi$")
plt.legend(loc="best", fontsize=8)

a = norm.cdf(mu1-mu2, 0, 1)
b = 1-norm.cdf(0, mu1 - mu2, 1)
print a
print b

plt.subplot(133)
a = 0.6
b = 0.5
x = np.arange (-50, 50, 0.1)
y = beta.pdf(x, a, b, scale=100, loc=-50)
plt.plot(x, y)
plt.show()
