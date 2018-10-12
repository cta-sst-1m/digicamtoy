import numpy as np
from scipy.special import factorial, loggamma, gammaln
from scipy.stats import rv_discrete


class Borel(rv_discrete):

    def __init__(self):

        super().__init__(a=1)

    def _pmf(self, k, mu_xt):

        return np.exp(self.logpmf(k, mu_xt))

    def logpmf(self, k, mu_xt):

        x = k * mu_xt
        k_factorial = gammaln(k + 1)
        log_prob = - x + (k - 1) * (np.log(k) + np.log(mu_xt)) - k_factorial

        return log_prob


class GeneralizedPoisson(rv_discrete):

    def _pmf(self, k, mu, mu_xt):

        prob = mu * (mu + k * mu_xt) ** (k - 1) / factorial(k)
        prob *= np.exp(-k * mu_xt - mu)

        return prob


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    mu = np.array([0.2, 0.1])[0]
    mu_xt = np.array([0.08, 0.1])[0]

    borel = Borel()
    data = borel.rvs(mu_xt=mu_xt, size=(10000, ))

    bins = np.arange(data.min(), data.max() + 1, 1)

    plt.figure()
    plt.hist(data, bins=bins, normed=True)
    plt.plot(bins, borel.pmf(bins, mu_xt=mu_xt))
    plt.plot(bins, np.exp(borel.logpmf(bins, mu_xt=np.array([mu_xt, mu_xt]))))

    generalized_poisson = GeneralizedPoisson()
    data = generalized_poisson.rvs(mu=mu,
                                   mu_xt=mu_xt, size=(10000,))

    bins = np.arange(data.min(), data.max() + 1, 1)

    plt.figure()
    plt.hist(data, bins=bins, normed=True)
    plt.plot(bins, generalized_poisson.pmf(bins, mu=mu, mu_xt=mu_xt))
    # plt.plot(bins, np.exp(borel.logpmf(bins, mu_xt=mu_xt)))

    plt.show()

