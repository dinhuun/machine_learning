import sys
from typing import Optional, Tuple

import numpy as np
from numpy import ndarray


def compute_centers(X: ndarray, labels: ndarray) -> ndarray:
    """
    computes center of each cluster of samples corresponding to each label in X
    :param X: samples
    :param labels: labels
    :return: centers
    """
    _, m = X.shape
    k = len(np.unique(labels))
    centers = np.empty((k, m))
    for i in range(k):
        label_mask = labels == i
        cluster = X[label_mask]
        center = np.mean(cluster, axis=0)
        centers[i] = center
    return centers


def compute_distance(u: ndarray, V: ndarray) -> ndarray:
    """
    computes distance between 1-d array and each row in 2-d array
    :param u: 1-d array
    :param V: 2-d array
    :return: distances
    """
    distances = np.linalg.norm(u - V, ord=2, axis=1)
    return distances


def compute_labels(X: ndarray, centers: ndarray) -> ndarray:
    """
    computes label of each sample in X, that is, which center is closest to sample
    :param X: samples
    :param centers: centers
    :return: labels
    """
    n = len(X)
    labels = np.full(n, -1, dtype=int)
    for i in range(n):
        x = X[i]
        distances = compute_distance(x, centers)
        label = np.argmin(distances)
        labels[i] = label
    return labels


def k_means(X: ndarray, k: int, n_iters: int = sys.maxsize, seed: Optional[int] = None) -> Tuple[ndarray, ndarray]:
    """
    implements k-means clustering using
        - naive algorithm, which does not guarantee to find the optimum
        - Forgy initialization method, which depends on seed quite a bit
    :param X: samples
    :param k: number of clusters
    :param n_iters: number of iterations in this naive algorithm
    :param seed: seed
    :return: centers and labels
    """
    if n_iters < 1:
        raise ValueError(f"number of iterations {n_iters} must be positive")

    n = len(X)
    if n <= k:
        return X, np.array(list(range(n)))

    np.random.seed(seed)
    init_indices = np.random.choice(n, k, replace=False)
    init_centers = X[init_indices]
    i = 0
    while i < n_iters:
        labels = compute_labels(X, init_centers)
        centers = compute_centers(X, labels)
        try:
            np.testing.assert_array_almost_equal(init_centers, centers)
            break
        except AssertionError:
            init_centers = centers
            i += 1
    return centers, labels
