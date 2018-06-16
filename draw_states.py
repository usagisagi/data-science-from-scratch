from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


def load_text():
    text = ""
    with open("states.txt", encoding="utf-8") as f:
        text = "".join(f)

    return text


def extract_corner(text):
    soup = BeautifulSoup(text, 'html5lib')
    state_list = soup.find_all('state')
    # List[List[(lat, lng)]] stateごとのlat, lngのリスト
    point_list = [
        [(float(point["lat"]), float(point["lng"])) for point in state.find_all("point")]
        for state in state_list
    ]

    return point_list


def draw_points(points_list):
    for points in points_list:
        for point_b, point_a in zip(points, points[1:]):
            plt.plot((point_b[1], point_a[1]), (point_b[0], point_a[0]),
                     color='black',
                     linewidth=1)


def draw_states():
    points_list = extract_corner(load_text())
    draw_points(points_list)


if __name__ == '__main__':
    pass
