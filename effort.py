from scipy.stats import norm, beta
from trueskill.mathematics import Gaussian
from matplotlib import pyplot as plt
import numpy as np

g1 = Gaussian(mu=5, sigma=1)
g2 = Gaussian(mu=2, sigma=1)

e1 = Gaussian(mu=3, sigma=1)
e2 = Gaussian(mu=1, sigma=1)
skill_diff = Gaussian(mu=g1.mu-g2.mu, sigma=np.sqrt(g1.sigma**2 + g2.sigma**2))
e_diff = Gaussian(mu=e1.mu-e2.mu, sigma=np.sqrt(e1.sigma**2 + e2.sigma**2))
skill_effort = skill_diff * e_diff

domain = np.linspace(-10, 10, 200)
y = [norm.pdf(x) for x in domain]

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(domain, [norm.pdf(x, g1.mu, g1.sigma) for x in domain], label="$P1$")
plt.plot(domain, [norm.pdf(x, g2.mu, g2.sigma) for x in domain], label="$P2$")
plt.plot(domain, [norm.pdf(x, e1.mu, e1.sigma) for x in domain], label="$E1$")
plt.plot(domain, [norm.pdf(x, e2.mu, e2.sigma) for x in domain], label="$E2$")
plt.legend(loc="best", fontsize=8)

plt.subplot(132)
plt.plot(domain, [norm.pdf(x, e_diff.mu, e_diff.sigma) for x in domain], label="$effortdiff$")
plt.plot(domain, [norm.pdf(x, skill_diff.mu, skill_diff.sigma) for x in domain], label="$skilldiff$")
plt.plot(domain, [norm.pdf(x, skill_effort.mu, skill_effort.sigma) for x in domain], label="$skill + effort$")
plt.legend(loc="best", fontsize=8)

plt.subplot(133)
plt.plot(domain, [norm.cdf(x, skill_diff.mu, skill_diff.sigma) for x in domain], label="$skilldiff$")
plt.plot(domain, [norm.cdf(x, skill_effort.mu, skill_effort.sigma) for x in domain], label="$skill + effort$")
plt.legend(loc="best", fontsize=8)
plt.show()
