"""Pure 2D geometry helpers for line-crossing detection (standard library only).

All points are ``(x, y)`` tuples in image coordinates (x to the right, y down).
"""

from __future__ import annotations

from typing import Tuple

Point = Tuple[float, float]


def ccw(a: Point, b: Point, c: Point) -> bool:
    """Return True if points ``a``, ``b``, ``c`` are in counter-clockwise order.

    The comparison is strict, so collinear points return False.

    Args:
        a: First point.
        b: Second point.
        c: Third point.

    Returns:
        Whether the turn ``a -> b -> c`` is counter-clockwise in the
        mathematical orientation of the coordinate values.
    """
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Return True if segment ``ab`` and segment ``cd`` intersect.

    Uses the classic CCW orientation test: the endpoints of each segment must
    lie on opposite sides of the other segment. Unlike an infinite-line test,
    a trajectory passing beside the end of the drawn segment is not counted.

    Args:
        a: First endpoint of the counting line.
        b: Second endpoint of the counting line.
        c: Previous position of the object.
        d: Current position of the object.

    Returns:
        True if the two segments cross.
    """
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def side_of_line(a: Point, b: Point, p: Point) -> float:
    """Return the signed 2D cross product ``(b - a) x (p - a)``.

    In image coordinates (y pointing down) a positive value means ``p`` is on
    the right-hand side of the directed line ``a -> b`` as seen on screen, a
    negative value means the left-hand side, and zero means it is on the line.

    Args:
        a: Start of the directed line.
        b: End of the directed line.
        p: Point to classify.

    Returns:
        The signed cross product.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segment_misses_frame(a: Point, b: Point, width: int, height: int) -> bool:
    """Return True if segment ``ab`` lies completely outside a ``width`` x ``height`` frame.

    Uses Liang-Barsky clipping against the pixel area ``[0, width-1] x [0, height-1]``.

    Args:
        a: First endpoint.
        b: Second endpoint.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        True if no part of the segment is inside the frame.
    """
    x0, y0 = a
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, width - 1 - x0), (-dy, y0), (dy, height - 1 - y0)):
        if p == 0:
            if q < 0:
                return True
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return True
    return False
