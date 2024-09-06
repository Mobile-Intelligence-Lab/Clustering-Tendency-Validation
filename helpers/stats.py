import numpy as np
from numpy.random import uniform
from random import sample
from math import isnan

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

DIAMETER_METHODS = ['mean_cluster', 'farthest']
CLUSTER_DISTANCE_METHODS = ['nearest', 'farthest']


def hopkins(X):
    X = np.array(X)
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2,
                                    return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X[rand_X[j]].reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = 0
    if isnan(H) or (sum(ujd) + sum(wjd)) == 0:
        H = 0
    else:
        H = sum(ujd) / (sum(ujd) + sum(wjd))

    return H


def inter_cluster_distances(labels, distances, method='nearest'):
    if method not in CLUSTER_DISTANCE_METHODS:
        raise ValueError(
            'method must be one of {}'.format(CLUSTER_DISTANCE_METHODS))

    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                    (not farthest and
                     distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                    or
                    (farthest and
                     distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
    if method not in DIAMETER_METHODS:
        raise ValueError('method must be one of {}'.format(DIAMETER_METHODS))

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn(data, labels, diameter_method='farthest',
         cdist_method='nearest'):
    """
    Dunn index for cluster validation (larger is better).

    .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
    """

    distances = euclidean_distances(data)
    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameter(labels, distances, diameter_method))

    return min_distance / max_diameter


# Extrinsic metrics (External cluster validation)
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def rand_index(y_true, y_pred):
    return metrics.rand_score(y_true, y_pred)


def adjusted_rand_index(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)


def nmi(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)


# Intrinsic metrics (Internal cluster validation)
def silhouette_coefficient(x, y):
    return metrics.silhouette_score(x, y)


def dunn_index(x, y):
    return dunn(x, y, diameter_method='farthest', cdist_method='nearest')


def db_index(x, y):
    return metrics.davies_bouldin_score(x, y)


def ch_index(x, y):
    return metrics.calinski_harabasz_score(x, y)


from collections import Counter


def cluster_acc(y_true, y_pred):
    per_class_accuracies = {}
    selected_cls = []
    for idx, cls in enumerate(list(set(y_true))):
        idx = np.nonzero(y_true == cls)[0]
        pred_items = y_pred[idx]
        ctr = Counter(pred_items.ravel())
        most_common_value, _ = ctr.most_common(1)[-1]
        common_idx = 1
        while most_common_value in selected_cls:
            common_idx += 1
            most_common_value, _ = ctr.most_common(common_idx)[-1]
        pred_counts = len(pred_items[pred_items == most_common_value])
        selected_cls.append(most_common_value)

        misc = (len(idx) - pred_counts) / len(idx)
        per_class_accuracies[cls] = misc
    acc = 1 - np.max(list(per_class_accuracies.values()))
    return acc


def accuracy(labels_true, labels_pred):
    return cluster_acc(labels_true, labels_pred)
