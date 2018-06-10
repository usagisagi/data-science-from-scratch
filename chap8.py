from functools import partial

import numpy as np
import matplotlib.pyplot as plt


def sum_of_squares(v):
    return np.sum(v_i ** 2 for v_i in v)


def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h


def derivative(x):
    return 2 * x


func = lambda x: x ** 2

xs = np.arange(-10, 10)

derivate_estimate = partial(difference_quotient, f=func, h=1e-4)

plt.scatter(xs, [derivative(x) for x in xs])
plt.scatter(xs, [derivate_estimate(x=x) for x in xs])


def partial_difference_quotient(f, v, i, h=1e-4):
    # vのi番目の偏微分を求める
    # i番目にhを加える
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=1e-4):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


def step(v, direction, step_size):
    # directionは-1
    return [v_i + step_size * direction_i for v_i, direction_i in zip(v, direction)]
