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


def test_segment_misses_frame():
    from line_counter.geometry import segment_misses_frame

    assert not segment_misses_frame((10, 10), (50, 50), 100, 80)
    assert not segment_misses_frame((-50, 40), (150, 40), 100, 80)   # passes through the frame
    assert not segment_misses_frame((-5, -5), (5, 5), 100, 80)       # one end inside
    assert segment_misses_frame((200, 10), (300, 50), 100, 80)       # right of the frame
    assert segment_misses_frame((-30, -30), (-10, -5), 100, 80)      # above-left
    assert segment_misses_frame((0, 500), (100, 500), 100, 80)       # below
    assert segment_misses_frame((-50, 100), (50, 200), 100, 80)      # bbox overlaps only in x, line below


def test_segment_misses_frame_diagonal_corner_case():
    from line_counter.geometry import segment_misses_frame

    # bounding box touches the frame but the segment cuts the corner outside of it
    assert segment_misses_frame((-100, 50), (50, 200), 100, 80)
    assert not segment_misses_frame((-100, 50), (50, 20), 100, 80)


def test_readme_direction_example():
    # README: a horizontal line drawn left to right counts a downward move as IN.
    from line_counter.counter import CountingLine

    line = CountingLine("L", (0, 50), (100, 50))
    assert line.update(1, "p", (50, 40), (50, 60)) == "IN"
    assert line.update(2, "p", (50, 60), (50, 40)) == "OUT"
    # the same line drawn right to left swaps the labels
    rev = CountingLine("R", (100, 50), (0, 50))
    assert rev.update(1, "p", (50, 40), (50, 60)) == "OUT"


def test_vertical_line_direction():
    from line_counter.counter import CountingLine

    line = CountingLine("V", (50, 0), (50, 100))  # drawn top -> bottom
    assert line.update(1, "p", (40, 50), (60, 50)) == "OUT"  # moving right = to the left of the arrow
    assert line.update(2, "p", (60, 50), (40, 50)) == "IN"
