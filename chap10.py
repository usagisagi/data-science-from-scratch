from collections import Counter
from chap6 import inverse_normal_cdf
import matplotlib.pyplot as plt
import numpy as np
import dateutil.parser
import csv
import datetime

np.random.seed(42)


def bucketize(point, bucket_size):
    """小数点以下を切り捨てて、揃える"""
    return bucket_size * np.floor(point / bucket_size)


def make_histogram(points, bucket_size):
    return Counter(bucketize(point, bucket_size) for point in points)


def plot_histogram(points, bucket_size, title=""):
    hist_data = make_histogram(points, bucket_size)
    plt.bar(list(hist_data.keys()), list(hist_data.values()), width=bucket_size)
    plt.show()


def random_normal():
    return inverse_normal_cdf(np.random.rand())


def make_matrix(num_rows, num_columns, f):
    return np.array([[f(i, j) for i in range(num_rows)] for j in range(num_columns)])


def correlation_matrix(data: np.ndarray):
    _, num_columns = data.shape

    def matrix_entry(i, j):
        return correlation(data[:, i], data[:, j])

    return make_matrix(num_columns, num_columns, matrix_entry)


def make_scatterplot_matrix():
    # first, generate some random data

    num_points = 100

    def random_row():
        row = [None, None, None, None]
        row[0] = random_normal()
        row[1] = -5 * row[0] + random_normal()
        row[2] = row[0] + row[1] + 5 * random_normal()
        row[3] = 6 if row[2] > -2 else 0
        return row

    data = np.array([random_row() for _ in range(num_points)])

    _, num_columns = data.shape
    fig, ax = plt.subplots(num_columns, num_columns)
    for i in range(num_columns):
        for j in range(num_columns):

            if i != j:
                ax[i][j].scatter(data[:, j], data[:, i], marker='.')
            else:
                ax[i][j].annotate("series" + str(i), (0.5, 0.5),
                                  xycoords='axes fraction',
                                  ha='center',
                                  va='center')
                if i < num_columns - 1: ax[i][j].xaxis.set_visible(False)
                if j > 0: ax[i][j].yaxis.set_visible(False)

    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())
    plt.show()


def parse_row(input_row, parsers):
    return [try_or_none(parser)(value) if parser is not None else value
            for (value, parser) in zip(input_row, parsers)]


def try_or_none(f):
    """
    ラッピングした関数を返す
    :param f:
    :return:
    """

    def f_or_none(x):
        try:
            return f(x)
        except:
            return None

    return f_or_none


def load_stocks(name="stocks.txt"):
    ret_list = []
    with open(name, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            ret_list.append(
                {
                    "symbol": r['symbol'],
                    'date:': datetime.datetime.strptime(r['date'], '%Y-%m-%d'),
                    'closing_price': float(r['closing_price'])
                }
            )

    return ret_list


if __name__ == '__main__':
    uniform = [200 * np.random.random() - 100 for _ in range(100000)]
    plot_histogram(uniform, 10)

    # random.rand()は0-1までの一様分布
    # 一様分布の値列(rand())を投入して、標準分布の逆
    normal = [57 * inverse_normal_cdf(np.random.rand()) for i in range(100000)]
    plot_histogram(normal, 10)

    xs = [random_normal() for _ in range(1000)]
    ys1 = [x + random_normal() / 2 for x in xs]
    ys2 = [-x + random_normal() / 2 for x in xs]

    plt.scatter(xs, ys1, marker=".")
    plt.scatter(xs, ys2, marker=".")
    plt.show()

    from chap5 import correlation

    correlation(ys1, ys2)
    correlation(xs, ys1)
    correlation(xs, ys2)

    max_appl_price = max([d['closing_price'] for d in load_stocks() if d['symbol'] == 'AAPL'])

    data = load_stocks()
    by_symbol = set([r['symbol'] for r in data])

    max_appl_price_in_a_symbol = lambda s : max([d['closing_price'] for d in data if d['symbol'] == s])
    max_appl_price_by_symbol = {}
    for s in by_symbol:
        max_appl_price_by_symbol[s] = max_appl_price_in_a_symbol(s)
