from typing import List, Union

import numpy as np
from more_itertools import divide

from machine_learning.classes import Point
from machine_learning.utils.utils_math import compute_triangle_area


def check_float_size(size: float):
    """
    checks whether float size is in (0, 1)
    """
    if size <= 0 or size >= 1:
        raise ValueError(f"down_size {size} must be in (0, 1)")


def check_int_size(size: float, n: int):
    """
    checks whether int size is in (0, n)
    """
    if size <= 0 or size >= n:
        raise ValueError(f"down_size {size} must be in (0, n_points)")


def ltob(points: List[Point], down_size: Union[int, float] = 0.1) -> List[Point]:
    """
    implements algorithm Largest Triangle One Bucket in Downsampling Time Series for Visual Representation
        * computes effective area from triplet (p, q, r) for each point q
        * divide points into buckets
        * picks point with largest effective area from each bucket
    :param points: points from which to sample
    :param down_size: number of samples
    :return: samples
    """
    n_points = len(points)
    if isinstance(down_size, float):
        check_float_size(down_size)
        down_size = round(n_points * down_size)
    check_int_size(down_size, n_points)

    points_areas = []
    for i in range(1, n_points - 1):
        p, q, r = points[i - 1], points[i], points[i + 1]
        area = compute_triangle_area(p, q, r)
        points_areas.append((q, area))

    buckets = [list(bucket) for bucket in divide(down_size - 2, points_areas)]
    samples = [
        max(bucket, key=lambda point_area: point_area[1])[0] for bucket in buckets
    ]
    samples.insert(0, points[0])
    samples.append(points[-1])
    return samples


def lttb(points: List[Point], down_size: Union[int, float] = 0.1) -> List[Point]:
    """
    implements algorithm Largest Triangle Three Buckets in Downsampling Time Series for Visual Representation
        * makes buckets of points from points
        * picks point q from each bucket with largest area from triplet (p, q, r) where
            * p is point picked from previous bucket
            * r is center of next bucket
    :param points: points from which to sample
    :param down_size: number of samples
    :return: samples
    """
    n_points = len(points)
    if isinstance(down_size, float):
        check_float_size(down_size)
        down_size = round(n_points * down_size)
    check_int_size(down_size, n_points)

    buckets = make_buckets(points, down_size)
    samples = pick_from_buckets(buckets)
    return samples


def make_buckets(points: List[Point], down_size: int) -> List[List[Point]]:
    """
    makes buckets of points from points
    :param points: points from which to sample
    :param down_size: number of samples
    :return: buckets of points
    """
    buckets = [list(bucket) for bucket in divide(down_size - 2, points[1:-1])]
    buckets.insert(0, [points[0]])
    buckets.append([points[-1]])
    return buckets


def pick_from_buckets(buckets: List[List[Point]]) -> List[Point]:
    """
    picks point q from each bucket with largest area from triplet (p, q, r) where
        * p is point picked from previous bucket
        * r is center of next bucket
    :param buckets: buckets
    :return: points from buckets
    """
    p = buckets[0][0]
    samples = [p]
    for i, bucket in enumerate(buckets[1:-1]):
        max_area = 0
        max_point = p
        mean_x_y = np.mean(np.array(buckets[i + 1]), axis=0)
        r = Point(*mean_x_y)
        for q in bucket:
            area = compute_triangle_area(p, q, r)
            if (
                area >= max_area
            ):  # equality in case all areas equal max_area 0, we want to update max_point p to q
                max_point = q
        samples.append(max_point)
        p = max_point
    samples.append(buckets[-1][0])
    return samples
