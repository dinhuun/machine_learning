from machine_learning.classes import Point
from machine_learning.utils.utils_math import compute_triangle_area


def test_compute_triangle_area():
    """
    tests compute_triangle_area()
    """
    assert compute_triangle_area(Point(), Point(), Point()) == 0.0
    assert compute_triangle_area(Point(0, 0), Point(0, 1), Point(1, 0)) == 0.5
    assert compute_triangle_area(Point(0, 0), Point(0.5, 1), Point(1, 0)) == 0.5
    assert compute_triangle_area(Point(-1, 0), Point(0, 1), Point(1, 0)) == 1.0
