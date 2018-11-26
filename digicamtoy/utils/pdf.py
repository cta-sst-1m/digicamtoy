import numpy as np
from scipy.special import factorial, gammaln
from scipy.stats import rv_discrete
import time


class Borel(rv_discrete):

    def __init__(self):

        super().__init__(a=1)

    def _pmf(self, k, mu_xt):

        x = mu_xt * k
        prob = np.exp(- x)
        prob *= x ** (k - 1)
        prob /= factorial(k)

        return

    # def _pmf(self, k, mu_xt):

    #    return np.exp(self.logpmf(k, mu_xt))

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

    n_pixels = 3
    n_photons = 1000

    mu = np.array([1])[0]
    mu_xt = np.linspace(0.0001, 1, num=n_pixels, endpoint=False)
    borel = Borel()

    t0 = time.time()

    data = borel.rvs(mu_xt=mu_xt, size=(n_photons, len(mu_xt))).T

    print(time.time() - t0)

    t0 = time.time()

    for i in range(n_pixels):

        borel.rvs(mu_xt[i], size=(n_photons, ))

    print(time.time() - t0)

    bins = np.arange(data.min(), data.max() + 1, 1)

    plt.figure()

    for i, m in enumerate(mu_xt):
        label = '$\mu_{XT} =$' + ' {}'.format(m)

        plt.hist(data[i], bins=bins, normed=True, alpha=0.3,
                 label=label)

        mean = np.mean(data[i])
        label += '\n Mean : {}'.format(mean)
        plt.axvline(mean, label=label)
        # plt.plot(bins, borel.pmf(bins, mu_xt=mu_xt[i]))
        # plt.plot(bins, np.exp(borel.logpmf(bins, mu_xt=mu_xt[i])))

    plt.legend(loc='best')

    generalized_poisson = GeneralizedPoisson()
    data = generalized_poisson.rvs(mu=mu,
                                   mu_xt=mu_xt, size=(n_photons, len(mu_xt)))

    bins = np.arange(data.min(), data.max() + 1, 1)

    plt.figure()

    for i, m in enumerate(mu_xt):

        label = '$\mu_{XT} =$' + ' {}'.format(m)
        plt.hist(data[:, i], bins=bins, normed=True, alpha=0.3,
                     label=label)
        mean = np.mean(data[:, i])
        label += '\n Mean : {}'.format(mean)
        plt.axvline(mean, label=label)
        # plt.plot(bins, generalized_poisson.pmf(bins, mu=mu, mu_xt=mu_xt[0]))
    # plt.plot(bins, np.exp(borel.logpmf(bins, mu_xt=mu_xt)))

    plt.legend()
    plt.show()

