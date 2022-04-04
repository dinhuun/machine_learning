import numpy as np

from machine_learning.strings import label_str, level_str, range_str
from machine_learning.utils.utils_mapper import (
    get_vertex_level,
    index_cubes,
    link_vertices,
    make_vertex_ID,
    reshape_array,
)

array_000 = np.array([0, 0, 0])
array_001 = np.array([0, 0, 1])
array_010 = np.array([0, 1, 0])
array_011 = np.array([0, 1, 1])
array_100 = np.array([1, 0, 0])
array_101 = np.array([1, 0, 1])
array_110 = np.array([1, 1, 0])
array_111 = np.array([1, 1, 1])
arrays = [
    array_000,
    array_001,
    array_010,
    array_011,
    array_100,
    array_101,
    array_110,
    array_111,
]

level = 0
range_ = np.array([0.0, 1.0])
label = 1
vertex_ID = f"{level_str}_{level}_{range_str}_[0. 1.]_{label_str}_{label}"

np.random.seed(0)
X = np.random.rand(10)


def test_get_vertex_level():
    """
    tests get_vertex_level()
    """
    assert get_vertex_level(vertex_ID) == level


def test_index_cubes():
    """
    tests index_cubes()
    """
    n_intervals = 2
    n_cols = 3
    cube_indices = index_cubes(n_intervals, n_cols)
    np.testing.assert_equal(cube_indices, arrays)


def test_link_vertices():
    """
    tests link_vertices()
    """
    vertices = {"a": [0, 1, 2], "b": [1, 2, 3], "c": [3, 4, 5], "d": [6, 7, 8]}
    sources = [0, 1]  # indices of "a", "b" in ["a", "b", "c", "d"]
    targets = [1, 2]  # indices of "b", "c" in ["a", "b", "c", "d"]
    values = [2, 1]  # sizes of intersections [1, 2] and [3]
    edge_sources, edge_targets, edge_values = link_vertices(vertices)
    assert edge_sources == sources
    assert edge_targets == targets
    assert edge_values == values


def test_make_vertex_ID():
    """
    tests make_vertex_ID()
    """
    assert make_vertex_ID(level, range_, label) == vertex_ID


def test_reshape_array():
    """
    tests reshape_array()
    """
    assert X.ndim == 1
    X_reshaped = reshape_array(X)
    assert X_reshaped.ndim == 2
