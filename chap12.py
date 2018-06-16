from collections import Counter
import matplotlib.pyplot as plt
from chap8 import distance
from chap12_data import cities
from draw_states import draw_states
import numpy as np


def raw_majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner


def majority_vote(labels):
    # labelsは近いものから遠いものにソートする
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])


def knn_classify(k, labeled_points, new_point):
    """point (point, label)のペア"""

    by_distance = sorted(labeled_points,
                         key=lambda elem: distance(elem[0], new_point))

    # 近い順にk個の点を取り出す
    k_nearest_labels = [label for _, label in by_distance[:k]]

    return majority_vote(k_nearest_labels)


def plot_favorite_programing_languages(cities):
    import seaborn as sns
    sns.set()
    draw_states()

    plots = {"Java": ([], []), "Python": ([], []), "R": ([], [])}

    for log, lat, language in cities:
        plots[language][0].append(log)
        plots[language][1].append(lat)

    for language, (x, y) in plots.items():
        plt.scatter(x, y, label=language, s=15)

    plt.legend(loc=0)
    plt.axis([-130, -60, 20, 55])
    plt.show()


def find_k():
    for k in [1, 3, 5, 7]:
        num_correct = 0
        labeled_cities = [((log, lat), language) for log, lat, language in cities]
        for city in labeled_cities:
            location, actual_language = city
            other_cities = [other_city for other_city in labeled_cities if other_city != city]

            predicted_language = knn_classify(k, other_cities, location)

            if predicted_language == actual_language:
                num_correct += 1

        print(k, "neighbor[s]", num_correct, "correct out of", len(labeled_cities))


def grid_k_means(k):
    labeled_cities = [((log, lat), language) for log, lat, language in cities]
    grid_points = [(log, lat) for log in range(-130, -60) for lat in range(20, 55)]
    predicted_language = [knn_classify(k, labeled_cities, grid_point) for grid_point in grid_points]
    return [(point[0], point[1], language) for point, language in zip(grid_points, predicted_language)]


def random_point(dim):
    return np.array([abs(np.random.randn()) for _ in range(dim)])


def random_distances(dim, num_pairs):
    return np.array(
        [np.linalg.norm(random_point(dim) - random_point(dim))
         for _ in range(num_pairs)]
    )


def calc_distances(max_dim, num_pairs):
    np.random.seed(42)
    avg_distances = []
    min_distances = []
    dim_range = range(1, max_dim)
    for dim in dim_range:
        distances = random_distances(dim, num_pairs)
        avg_distances.append(np.average(distances))
        min_distances.append(np.min(distances))
    return avg_distances, min_distances, dim_range


def plot_random_distance(max_dim, num_pairs):
    avg_distances, min_distances, dim_range = calc_distances(max_dim, num_pairs)

    import seaborn as sns
    sns.set()

    plt.plot(dim_range, avg_distances, label="average")
    plt.plot(dim_range, min_distances, label="min")
    min_avg_ratio = [min_dist / avg_dist for min_dist, avg_dist in zip(min_distances, avg_distances)]
    plt.plot(dim_range, min_avg_ratio)
    plt.legend()
    plt.show()


def plot_min_avg_ratio(max_dim, num_pairs):
    avg_distances, min_distances, dim_range = calc_distances(max_dim, num_pairs)
    import seaborn as sns
    sns.set()
    min_avg_ratio = [min_dist / avg_dist for min_dist, avg_dist in zip(min_distances, avg_distances)]
    plt.plot(dim_range, min_avg_ratio)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_favorite_programing_languages(cities)
    find_k()

    predicted_data = grid_k_means(3)
    plot_favorite_programing_languages(predicted_data)


