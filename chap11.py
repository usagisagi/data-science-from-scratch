import numpy as np

np.random.seed(0)


def split_data(data: np.ndarray, prob=0.66):
    """データを prob : (1-prob)に分割する"""
    buf_data = data.copy()
    np.random.shuffle(buf_data)
    num = int(np.floor(buf_data.shape[0] * prob))

    return buf_data[:num], buf_data[num:]


def train_test_split(x, y, test_pct):
    data = np.array(list(zip(x, y)))
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = list(zip(*train))
    x_test, y_test = list(zip(*test))
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def accuracy(tp, fp, fn, tn):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


def precision(tp, fp, fn, tn):
    return tp / (tp + fp)


def recall(tp, fp, fn, tn):
    return tp / (tp + fn)


def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)


if __name__ == '__main__':
    nums = (70, 4930, 13930, 981070)
    accuracy(*nums)
    precision(*nums)
    recall(*nums)
    f1_score(*nums)