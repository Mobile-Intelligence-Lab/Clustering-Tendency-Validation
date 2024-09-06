import numpy as np
from sklearn import datasets
import random

np.random.seed(0)


def gen_grid(n_points_per_row=9):
    data = np.meshgrid(np.linspace(1, n_points_per_row, n_points_per_row),
                       np.linspace(1, n_points_per_row, n_points_per_row))
    data = list(zip(data[0].reshape(-1), data[1].reshape(-1)))
    data = np.array(data)
    return data, np.zeros(len(data))


def gen_random(n_samples=150):
    data = np.array([(random.random() * 2.0, random.random() * 2.0) for _ in range(n_samples)])
    data = np.array(data)
    return data, np.zeros(len(data))


def gen_circles(n=1, n_samples=150):
    theta = np.linspace(0, 2 * np.pi, n_samples)
    radiuses = [i + 1 for i in range(n)]
    points = []
    labels = []
    for i, radius in enumerate(radiuses):
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        points_r = list(map(list, zip(x, y)))
        points += points_r
        labels += [i] * len(points_r)
    return np.asarray(points), np.asarray(labels)


def get_blobs(n_samples=150):
    random_state = 170
    X, original_labels = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    x = np.dot(X, transformation)
    return x, original_labels


def gen_cos_sin():
    x = [[i, np.cos(i)] for i in np.arange(-5, 5, .3)]
    x = x + [[i, np.sin(i)] for i in np.arange(-5, 5, .3)]
    original_labels = [0] * int(len(x) / 2) + [1] * int(len(x) / 2)
    x = np.array(x)
    original_labels = np.array(original_labels)
    return x, original_labels


def iris_dataset():
    iris = datasets.load_iris()
    return iris['data'], iris['target']


def digits_dataset():
    digits = datasets.load_digits()
    return digits['data'], digits['target']


def get_dataset(name):
    """
    Fetches dataset by name.
    """
    if name.lower() == "grid":
        return gen_grid()
    if name.lower() == "random":
        return gen_random()
    elif name.lower() == "iris":
        return iris_dataset()
    elif name.lower() == "digits":
        return digits_dataset()
    elif name.lower() == "blobs":
        return get_blobs()
    elif name.lower() == "sine/cosine":
        return gen_cos_sin()
    elif name.lower() == "2 moons":
        return datasets.make_moons(n_samples=150, noise=0.05)
    elif name.lower() == "circle":
        return gen_circles(n=1)
    elif name.lower() == "2 circles":
        return gen_circles(n=2)
    elif name.lower() == "5 circles":
        return gen_circles(n=5)
    else:
        raise ValueError("Dataset not found.")

