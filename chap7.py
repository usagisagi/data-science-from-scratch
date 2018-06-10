import numpy as np
import scipy as sp
from scipy import special
import matplotlib.pyplot as plt


def normal_approximation_to_binomial(n, p):
    mu = p * n
    sigma = np.sqrt(p * (1 - p) * n)
    return mu, sigma


def normal_cdf(x, mu=0, sigma=1):
    return 1 + sp.math.erf((x - mu) / (np.sqrt(2) * sigma))


# 閾値を下回る確率
normal_probability_below = normal_cdf


def normal_probability_above(lo, mu=0, sigma=1):
    """
    閾値を上回る確率
    :param lo:
    :param mu:
    :param sigma:
    :return:
    """
    return 1 - normal_cdf(lo, mu, sigma)


def normal_probability_between(hi, lo, mu, sigma):
    """
    閾値の間にある確率
    :param hi:
    :param lo:
    :param mu:
    :param sigma:
    :return:
    """
    return normal_probability_below(hi, mu, sigma) - normal_probability_below(lo, mu, sigma)


def normal_probability_outside(hi, lo, mu, sigma):
    return 1 - normal_probability_between(hi, lo, mu, sigma)


def B(alpha, beta):
    return special.gamma(alpha) * special.gamma(beta) / special.gamma(alpha + beta)


def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)


xs = np.arange(0.0, 1.0 + 0.001, 0.001)
plt.plot(xs, [beta_pdf(x, 20, 20) for x in xs])
plt.plot(xs, [beta_pdf(x, 4, 10) for x in xs])
plt.plot(xs, [beta_pdf(x, 14, 10) for x in xs])
plt.plot(xs, [beta_pdf(x, 4, 16) for x in xs])
