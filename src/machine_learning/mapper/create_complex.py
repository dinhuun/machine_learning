"""
This module implements Mapper method in topological data analysis.
Given a shape X with its image and other hyperparameters, this method constructs an abstract simplicial complex C
that approximates X.
If one does well, C is homotopic to X (as a circle is homotopic to a cylinder).
If one does really well, C is homeomorphic to X (as a circle is homeomorphic to an ellipse.)
How to use this information is up to us.
"""

from collections import defaultdict
from typing import Any, Dict

import numpy as np
from numpy import ndarray
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from machine_learning.classes import Complex_
from machine_learning.utils.utils_mapper import (
    get_vertex_level,
    index_cubes,
    link_vertices,
    make_vertex_ID,
    reshape_array,
)


def create_complex(
    X: ndarray,
    Y: ndarray,
    n_intervals: int = 5,
    overlap_ratio: float = 0.1,
    cluster_class: Any = None,
    cluster_params: Dict = None,
    verbose: bool = False,
) -> Complex_:
    """
    creates 1-skeleton of topological data analysis Mapper complex.
    See machine_learning/papers/topological_data_analysis.pdf for details
    :param X: data
    :param Y: image of data by some map
    :param n_intervals: number of intervals to divide each dimension of image by
    :param overlap_ratio: how much image subsets overlap
                          example: interval_i + overlap_ratio * interval_width and interval_i_plus_1
    :param cluster_class: any cluster class with method fit() and attribute labels_
    :param cluster_params: cluster class's initial params
    :param verbose: whether to index of empty cubes
    :return: complex
    """
    X = reshape_array(X)
    Y = reshape_array(Y)
    Y = MinMaxScaler().fit_transform(Y)
    n_rows, n_cols = Y.shape

    if cluster_class is None:
        cluster_class = DBSCAN

    mins = np.min(Y, axis=0)
    maxs = np.max(Y, axis=0)
    interval_widths = (maxs - mins) / n_intervals
    epsilons = overlap_ratio * interval_widths

    indices = np.arange(n_rows)

    cube_indices = index_cubes(n_intervals, n_cols)

    # go through each cube, get ys in cube, get its xs and cluster that into vertices
    vertices = defaultdict(list)
    for level, cube_index in enumerate(cube_indices):
        # boolean mask for xs/ys with ys in cube
        mask = np.all(
            (Y >= mins + cube_index * interval_widths)
            & (Y < mins + (cube_index + 1) * interval_widths + epsilons),
            axis=1,
        )
        ys_in_cube = Y[mask]

        # if ys in cube is nonempty, cluster its xs
        if len(ys_in_cube) > 0:
            indices_in_cube = indices[mask].tolist()
            xs_in_cube = X[mask]
            if cluster_params is None:
                clusterer = cluster_class()
            else:
                clusterer = cluster_class(**cluster_params)
            clusterer.fit(xs_in_cube)

            # add index of each x not labeled as noise to vertex corresponding to its cluster
            for x_index, label in zip(indices_in_cube, clusterer.labels_):
                if label != -1:
                    range_ = mins + cube_index * interval_widths
                    vertex_ID = make_vertex_ID(level, range_, label)
                    vertices[vertex_ID].append(x_index)
        else:
            if verbose:
                print(f"cube {level} is empty")

    vertex_IDs = list(vertices.keys())
    vertex_levels = [get_vertex_level(vertex_ID) for vertex_ID in vertex_IDs]
    vertex_sizes = [len(value) for value in vertices.values()]
    edge_sources, edge_targets, edge_values = link_vertices(vertices)
    complex_ = Complex_(
        vertices,
        vertex_IDs,
        vertex_levels,
        vertex_sizes,
        edge_sources,
        edge_targets,
        edge_values,
    )
    return complex_
