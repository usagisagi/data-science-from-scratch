from collections import Counter, defaultdict
from functools import reduce, partial
from typing import Iterable, Dict, List

from chap6 import inverse_normal_cdf
import matplotlib.pyplot as plt
import numpy as np
import dateutil.parser
import csv
import datetime
from chap10_data import X
from chap8 import maximize_stochastic

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
    return np.array([[f(i, j) for j in range(num_columns)] for i in range(num_rows)])


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


def load_stocks(name="stocks.txt") -> List:
    ret_list = []
    with open(name, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            ret_list.append(
                {
                    "symbol": r['symbol'],
                    'date': datetime.datetime.strptime(r['date'], '%Y-%m-%d'),
                    'closing_price': float(r['closing_price'])
                }
            )

    return ret_list


def picker(field_name: str):
    return lambda r: r[field_name]


def pluck(field_name: str, rows: Dict) -> Iterable:
    """辞書 -> 指定した列のリストに変換する"""
    return map(picker(field_name), rows)


def group_by(grouper, rows: Iterable, val_trans=None) -> Dict:
    """
    :param grouper: rowsの各要素からキーを取り出す関数
    :param rows: 行のiter
    :param val_trans: keyで取り出したリストを値に変換する関数
    :return: 辞書型

    """

    # defaultdict(default_factory)
    # 辞書にキーが存在しなければ、default_factoryで初期化
    grouped = defaultdict(list)
    for row in rows:
        grouped[grouper(row)].append(row)
    # この時点でキーはDict

    if val_trans is None:
        return grouped
    else:
        return {k: val_trans(row) for k, row in grouped.items()}


def percent_price_change(yesterday, today):
    return today['closing_price'] / yesterday['closing_price'] - 1


def day_over_day_changes(grouped_rows):
    """day"""
    # ソート
    ordered = sorted(grouped_rows, key=picker("date"))
    # 前日比を計算する
    return [{
        "symbol": today["symbol"],
        "date": today["date"],
        "change": percent_price_change(yesterday, today)
    } for (yesterday, today) in zip(ordered, ordered[1:])]


def combine_pct_changes(pct_change1, pct_change2):
    """パーセンテージの合成"""
    return (1 + pct_change1) * (1 + pct_change2) - 1


def overall_change(changes):
    return reduce(combine_pct_changes, pluck("change", changes))


def scale(data_matrix: np.ndarray):
    """列ごとの平均値と行順偏差を返す"""
    means = np.mean(data_matrix, axis=0)
    stdevs = np.std(data_matrix, axis=0)
    return means, stdevs


def rescale(data_matrix: np.ndarray):
    """平均0、標準偏差が1になるようにスケールを変換する"""

    means, stdevs = scale(data_matrix)

    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i, j] - means[j]) / stdevs[j]
        else:
            # 　標準偏差が0の場合は動かない
            return data_matrix[i, j]

    shape = data_matrix.shape
    return make_matrix(shape[0], shape[1], rescaled)


def de_mean_matrix(A: np.ndarray):
    """各列の平均を0にする"""
    col_means, _ = scale(A)
    # 各列の値を列平均から引く
    return make_matrix(A.shape[0], A.shape[1], lambda i, j: A[i, j] - col_means[j])


def direction(w):
    """大きさ1のベクトルを返す"""
    mag = np.linalg.norm(w)
    return w / mag


def directional_variance_i(x_i, w):
    """w方向のx_iの各店の2乗を求める"""
    # wで線形変換した後に二乗する
    # 平均が０なので2乗するだけでいい
    return np.dot(x_i, direction(w)) ** 2


def directional_variance(X, w):
    """w方向のデータの分散を求める"""
    return (directional_variance(x_i, w) for x_i in X)


def directional_variance_gradient_i(x_i, w):
    """w_i列の値がw方向にある分散に対する勾配"""

    # w方向で0からどれだけ離れているか
    projection_length = np.dot(x_i, direction(w))

    # x_iの各x,yに対し(0, 0)からwの投影距離の2倍を乗じる
    return np.array([2 * projection_length * x_ij for x_ij in x_i])


def directional_variance_gradient(X: np.ndarray, w: np.ndarray):
    """各点の勾配の和"""
    return np.sum(directional_variance_gradient_i(x_i, w) for x_i in X)


def first_principal_component(X: np.ndarray):
    guess = np.array([1 for _ in X[0]])  # 初期値, 各成分が1
    unscaled_maximizer = maximize_stochastic(
        lambda x_i, _, w: directional_variance_i(x_i, w),  # 各点に対しw方向の分散を計算してその和が最大になるようにする
        lambda x_i, _, w: directional_variance_gradient_i(x_i, w),  # 各点に対しw方向の勾配
        X,
        np.array([None for _ in X]),  # dummyのy
        guess  # 初期値
    )

    return direction(unscaled_maximizer)


def project(v, w):
    """点vをw方向に射影したベクトルを返す"""
    # w方向の長さ
    projection_length = np.dot(v, w)

    # w方向のprojection_lengthの長さ
    return w * projection_length


def remove_projection_from_vector(v, w):
    """vからw成分を取り除く"""
    return v - project(v, w)


def remove_projection(X, w):
    """Xの各成分に対してw成分を取り除く"""
    return np.array([remove_projection_from_vector(x_i, w) for x_i in X])


def principal_component_analysis(X, num_components):
    """主成分分析により、軸を決定する"""
    components = []
    for _ in range(num_components):
        component = first_principal_component(X)
        components.append(component)
        X = remove_projection(X, component)

    return np.array(components)


def transform_vector(v, components):
    return np.array([np.dot(v, w) for w in components])


def transform(X, components):
    """Xの各成分をcomponentsの軸に変換する"""
    return np.array([transform_vector(x_i, components) for x_i in X])


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
    max_appl_price_by_symbol = group_by(picker('symbol'),
                                        data,
                                        lambda r: max(pluck('closing_price', r)))
    changes_by_symbol = group_by(picker("symbol"), data, day_over_day_changes)
    # またsymbolに戻る
    all_changes = [change for changes in changes_by_symbol.values() for change in changes]

    min(all_changes, key=picker("change"))

    overall_change_by_month = group_by(lambda r: r['date'].month,
                                       all_changes,
                                       overall_change)
    plt.style.use('seaborn-whitegrid')
    plt.scatter(X[:, 0], X[:, 1])

    de_meaned_X = de_mean_matrix(X)
    w = first_principal_component(de_meaned_X)
    rm_X = remove_projection(de_meaned_X, w)

    plt.scatter(rm_X[:, 0], rm_X[:, 1])
    axis = principal_component_analysis(de_meaned_X, 2)