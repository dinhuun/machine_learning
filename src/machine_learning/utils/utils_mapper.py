from itertools import combinations, permutations
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray

from machine_learning.strings import label_str, level_str, range_str


def get_vertex_level(vertex_ID: str):
    """
    gets vertex level recorded in vertex_ID by make_vertex_ID()
    """
    level = int(vertex_ID.split("_")[1])
    return level


def index_cubes(n_intervals, n_cols):
    """
    multidimensionally indexes cubes
    example: see unit test
    :param n_intervals:
    :param n_cols:
    :return: multidimensional indices of cubes
    """
    cube_sides = list(range(n_intervals)) * n_cols
    side_permutations = permutations(cube_sides, n_cols)
    cube_indices = [np.array(i) for i in sorted(set(side_permutations))]
    return cube_indices


def link_vertices(
    vertices: Dict[str, List[int]]
) -> Tuple[List[int], List[int], List[int]]:
    """
    links two vertices' indices if vertices intersect
    :param vertices: vertices
    :return: edge sources, edge targets, and edge values
    """
    vertex_IDs = vertices.keys()
    IDs_indices = dict([(ID, index) for index, ID in enumerate(vertex_IDs)])
    ID_pairs = combinations(vertex_IDs, 2)
    edge_sources = []
    edge_targets = []
    edge_values = []
    for ID_0, ID_1 in ID_pairs:
        intersection = set(vertices[ID_0]) & set(vertices[ID_1])
        if intersection:
            edge_sources.append(IDs_indices[ID_0])
            edge_targets.append(IDs_indices[ID_1])
            edge_values.append(len(intersection))
    return edge_sources, edge_targets, edge_values


def make_vertex_ID(level: int, range_: ndarray, label: int) -> str:
    """
    makes ID for vertex based on its info
    :param level: vertex level
    :param range_: left endpoints of cube
    :param label: label of xs in vertex
    :return: vertex ID
    """
    vertex_ID = f"{level_str}_{level}_{range_str}_{range_}_{label_str}_{label}"
    return vertex_ID


def reshape_array(array: ndarray) -> ndarray:
    """
    reshapes array to 2d if it is 1d
    """
    if array.ndim == 1:
        array = array[:, None]
    return array
