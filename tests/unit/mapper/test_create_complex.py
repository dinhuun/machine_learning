import numpy as np

from machine_learning.mapper.create_complex import create_complex

n_samples = 1000
X = np.random.rand(n_samples)
Y = X
n_intervals = 5


def test_create_complex():
    """
    tests create_complex()
    """
    complex_ = create_complex(X, Y, n_intervals)
    # that there should be at least n_intervals vertices
    assert len(complex_.vertex_levels) >= n_intervals
    # that all vertices contain all samples
    samples = set().union(*complex_.vertices.values())
    assert len(samples) == n_samples
    # that some samples end up in more than one vertex
    assert np.sum(complex_.vertex_sizes) > n_samples
    # and so there is at least 1 edge
    assert len(complex_.edge_sources) > 0
    # and there are as many edge sources as edge targets as edge values
    assert (
        len(complex_.edge_sources)
        == len(complex_.edge_targets)
        == len(complex_.edge_values)
    )
