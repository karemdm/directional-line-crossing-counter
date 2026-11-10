"""Tests for line_counter.geometry (standard library only, pytest-compatible)."""

from line_counter.geometry import ccw, segments_intersect, side_of_line

A, B = (0, 0), (10, 0)  # horizontal counting line, directed left -> right


def test_ccw_orientation() -> None:
    assert ccw((0, 0), (10, 0), (5, 5)) is True
    assert ccw((0, 0), (10, 0), (5, -5)) is False
    assert ccw((0, 0), (10, 0), (5, 0)) is False  # collinear is not CCW


def test_crossing_through_middle() -> None:
    assert segments_intersect(A, B, (5, -5), (5, 5))
    assert segments_intersect(A, B, (5, 5), (5, -5))
    assert segments_intersect(A, B, (2, -3), (8, 4))  # diagonal crossing


def test_no_crossing_when_on_one_side() -> None:
    assert not segments_intersect(A, B, (5, 1), (5, 9))
    assert not segments_intersect(A, B, (5, -9), (5, -1))


def test_passing_beside_segment_end_is_not_a_crossing() -> None:
    # Would cross the infinite line, but not the drawn segment.
    assert not segments_intersect(A, B, (15, -5), (15, 5))
    assert not segments_intersect(A, B, (-3, -5), (-3, 5))


def test_parallel_and_collinear_segments() -> None:
    assert not segments_intersect(A, B, (0, 5), (10, 5))   # parallel
    assert not segments_intersect(A, B, (2, 0), (8, 0))    # collinear, overlapping
    assert not segments_intersect(A, B, (12, 0), (20, 0))  # collinear, disjoint


def test_touching_without_crossing() -> None:
    assert not segments_intersect(A, B, (-5, -5), (0, 0))  # ends on the line endpoint
    assert not segments_intersect(A, B, (5, -5), (5, 0))   # ends on the line
    assert not segments_intersect(A, B, (5, 0), (5, -5))   # starts on the line, leaves upward


def test_on_line_points_are_grouped_with_the_negative_side() -> None:
    # The strict CCW test treats a point exactly on the line like the negative
    # side, so these steps between "on the line" and the positive side count.
    assert segments_intersect(A, B, (5, 0), (5, 5))   # starts on the line, moves to positive
    assert segments_intersect(A, B, (5, 5), (5, 0))   # ends on the line, from positive


def test_side_of_line_sign() -> None:
    # Image coordinates: y grows downward, so (5, 5) is below the line.
    assert side_of_line(A, B, (5, 5)) > 0
    assert side_of_line(A, B, (5, -5)) < 0
    assert side_of_line(A, B, (5, 0)) == 0


def test_direction_flips_with_line_orientation() -> None:
    p = (5, 5)
    assert side_of_line(A, B, p) == -side_of_line(B, A, p)


def test_endpoint_touching_is_asymmetric() -> None:
    # Documented edge case of the strict CCW test at a line endpoint.
    assert segments_intersect(A, B, (5, 5), (0, 0))       # positive side -> endpoint
    assert not segments_intersect(A, B, (0, 0), (5, 5))   # endpoint -> positive side
