from collections import Counter
from random import random

import numpy as np
import matplotlib.pyplot as plt


def random_kid():
    return np.random.choice(['boy', 'girl'])


both_girls = 0
older_girl = 0
either_girl = 0

np.random.seed(42)

for i in range(10000):
    younger = random_kid()
    older = random_kid()

    if older == 'girl':
        older_girl += 1
    if younger == 'girl' and older == 'girl':
        both_girls += 1
    if younger == 'girl' or older == 'girl':
        either_girl += 1

print(f"P(both_girl|older_girl)={both_girls / older_girl}")
print(f"P(both_girl|either_girl)={both_girls / either_girl}")


def uniform_df(x):
    return 1 if 0 <= x < 1 else 0


plt.plot(np.arange(-1.0, 1.0, 0.1), [uniform_df(x) for x in np.arange(-1.0, 1.0, 0.1)])


def normal_pdf(x, mu=0, sigma=1.0):
    sqrt_two_pi = np.sqrt(2 * np.pi) * sigma
    exp = np.exp(- np.power(x - mu, 2) / (2 * sigma ** 2))
    return exp / sqrt_two_pi


xs = [x / 10.0 for x in range(-50, 50)]

plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--')
plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':')



def inverse_normal_cdf(p, mu=0, sigma=1, torerance=0.000001):
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, torerance=torerance)

    low_z, low_p = -10.0, 0.0
    hi_z, hi_p = 10.0, 1
    mid_z = (low_z + hi_z) / 2

    while hi_z - low_z > torerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_pdf(mid_z)

        if mid_p < p:
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            hi_z, hi_p = mid_z, mid_p
        else:
            break

    return mid_z


def bernoulli_trial(p):
    return 1 if random.random() < p else 0


def binomial(n, p):
    return sum(bernoulli_trial(p) for _ in range(n))


def make_hist(p, n, num_points):
    data = [binomial(n, p) for _ in range(num_points)]
    histogram = Counter(data)

    # 　正規分布を重ねる
    xs = [x for x in range(min(histogram.keys()), max(histogram.keys()) + 1)]
    ys = [normal_pdf(x, n*p, np.sqrt(n * p * (1 - p))) for x in xs]
    plt.plot(xs, ys)
    plt.bar(histogram.keys(), [y / num_points for y in histogram.values()])

make_hist(0.5, 100, 10000000)
