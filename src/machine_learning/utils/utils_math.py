from machine_learning.classes import Point


def compute_triangle_area(p: Point, q: Point, r: Point) -> float:
    """
    computes area of triangle by 3 points in a plane
    """
    term_0 = p.x * (q.y - r.y)
    term_1 = q.x * (r.y - p.y)
    term_2 = r.x * (p.y - q.y)
    return 0.5 * abs(term_0 + term_1 + term_2)
