import numpy as np
import pytest

from machine_learning.clustering.clusterer import compute_centers, compute_distance, compute_labels, k_means


X = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
n = len(X)


def test_compute_centers():
    """
    tests compute_centers()
    """
    labels = np.array([0, 0, 1, 1])
    centers = np.array([[0, 0], [0, 0]])
    np.testing.assert_equal(compute_centers(X, labels), centers)


def test_compute_distance():
    """
    tests compute_distance()
    """
    u = np.array([0, 1])
    V = np.array([[0, 0], [1, 0]])
    distances = np.array([1, np.sqrt(2)])
    np.testing.assert_equal(compute_distance(u, V), distances)


def test_compute_labels():
    """
    tests compute_labels()
    """
    centers = np.array([[-1, -1], [1, 1]])
    labels = np.array([0, 1, 0, 1])
    np.testing.assert_equal(compute_labels(X, centers), labels)


def test_k_means():
    """
    tests k_means()
    """
    with pytest.raises(ValueError):
        k_means(X, 2, 0)

    centers, labels = k_means(X, n)
    np.testing.assert_equal(centers, X)
    np.testing.assert_equal(labels, np.array(list(range(n))))
