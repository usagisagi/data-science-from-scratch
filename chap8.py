from functools import partial

import numpy as np
import matplotlib.pyplot as plt


def sum_of_squares(v):
    return np.sum(v_i ** 2 for v_i in v)


def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h


def derivative(x):
    return 2 * x


def partial_difference_quotient(f, v, i, h=1e-4):
    # vのi番目の偏微分を求める
    # i番目にhを加える
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=1e-4):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


def step(v, direction, step_size):
    # direction方向にstep_size移動する
    return np.array([v_i + step_size * direction_i for v_i, direction_i in zip(v, direction)])


def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]


def distance(v1, v2):
    return np.sum((np.array(v1) - np.array(v2)) ** 2)


def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')

    return safe_f


def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=1e-6):
    """
    関数の最小値を求める

    :param target_fn:
        最小点を求める関数
    :param gradient_fn:
        勾配を求める関数（微分）
    :param theta_0:
    :param tolerance:
    :return:
        最小点のthetaを返す
    """
    step_sizes = [10 ** x for x in range(-4, 3)]

    theta = theta_0  # theta_0は開始点
    target_fn = safe(target_fn)
    value = target_fn(theta)  # target_fnでvalueを求める

    while True:
        # 点thetaでの勾配を求める
        gradient = gradient_fn(theta)

        # 点thetaの候補を選択する。stepは移動する関数。
        next_thetas = [step(theta, gradient, -step_size) for step_size in step_sizes]

        # valueを最小にするものを選択する。
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_thetas)

        if abs(value - next_value) < tolerance:
            return theta

        theta, value = next_theta, next_value


def negate(f):
    return lambda *args, **kwargs: -f(*args, **kwargs)


def negate_all(f):
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]


def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=1e-7):
    return minimize_batch(negate(target_fn), negate_all(gradient_fn), theta_0, tolerance)


def in_random_order(data):
    indexes = [i for i in range(len(data))]
    np.random.shuffle(indexes)
    for i in indexes:
        yield data[i]


def minimize_stochastic(target_fn, gradient_fn, x: np.ndarray, y: np.ndarray, theta_0, alpha_0=0.01) -> np.ndarray:
    data = np.array(list(zip(x, y)))
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0

    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            iterations_with_no_improvement += 1
            alpha *= 0.9

        for x_i, y_i in in_random_order(data):
            gradient_i = np.array(gradient_fn(x_i, y_i, theta))
            theta = theta - gradient_i * alpha

    return min_theta


def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn), negate_all(gradient_fn), x, y, theta_0, alpha_0)


if __name__ == '__main__':
    func = lambda x: x ** 2

    xs = np.arange(-10, 10)

    derivate_estimate = partial(difference_quotient, f=func, h=1e-4)

    plt.scatter(xs, [derivative(x) for x in xs])
    plt.scatter(xs, [derivate_estimate(x=x) for x in xs])
    v = [np.random.randint(-10, 10) for i in range(3)]

    tolerance = 1e-10

    while True:
        gradient = sum_of_squares_gradient(v)
        next_v = step(v, gradient, -0.01)
        d = distance(next_v, v)
        if d < tolerance:
            break
        v = next_v

    print(v)
