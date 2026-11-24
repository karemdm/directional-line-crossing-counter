"""Tests for line_counter.counter and line_counter.lines_io (standard library only)."""

import json
import os
import tempfile

from line_counter.counter import IN, OUT, CountingLine, LineCounter, riding_ids
from line_counter.lines_io import lines_from_data, lines_to_data, load_lines, save_lines


def _horizontal(name: str = "Line_1") -> CountingLine:
    return CountingLine(name, (0, 0), (10, 0))


def test_direction_in_negative_to_positive() -> None:
    line = _horizontal()
    assert line.update(1, "Person", (5, -5), (5, 5)) == IN
    assert line.counts["Person"] == {IN: 1, OUT: 0}


def test_direction_out_positive_to_negative() -> None:
    line = _horizontal()
    assert line.update(1, "Car", (5, 5), (5, -5)) == OUT
    assert line.counts["Car"] == {IN: 0, OUT: 1}


def test_no_count_without_crossing() -> None:
    line = _horizontal()
    assert line.update(1, "Person", (5, 1), (5, 9)) is None
    assert line.counted_ids == set()


def test_same_id_counted_once_per_line() -> None:
    line = _horizontal()
    assert line.update(7, "Person", (5, -5), (5, 5)) == IN
    assert line.update(7, "Person", (5, 5), (5, -5)) is None  # crosses back: ignored
    assert line.update(7, "Car", (5, -5), (5, 5)) is None     # class flicker: still ignored
    assert line.counts["Person"] == {IN: 1, OUT: 0}
    assert "Car" not in line.counts


def test_different_ids_are_counted_separately() -> None:
    line = _horizontal()
    line.update(1, "Person", (5, -5), (5, 5))
    line.update(2, "Person", (6, -5), (6, 5))
    line.update(3, "Person", (7, 5), (7, -5))
    assert line.counts["Person"] == {IN: 2, OUT: 1}


def test_line_counter_tracks_previous_positions() -> None:
    counter = LineCounter([_horizontal()], ["Person", "Car"])
    assert counter.update(1, "Person", (5, -5)) == []  # first sighting: nothing to compare
    assert counter.update(1, "Person", (5, -1)) == []
    assert counter.update(1, "Person", (5, 3)) == [("Line_1", IN)]
    assert counter.update(1, "Person", (5, -3)) == []  # already counted on this line
    assert counter.summary() == {"Line_1": {"Person": {IN: 1, OUT: 0}, "Car": {IN: 0, OUT: 0}}}


def test_same_id_counted_on_each_line_independently() -> None:
    lines = [CountingLine("Line_1", (0, 0), (10, 0)), CountingLine("Line_2", (0, 10), (10, 10))]
    counter = LineCounter(lines)
    counter.update(4, "Car", (5, -5))
    assert counter.update(4, "Car", (5, 5)) == [("Line_1", IN)]
    assert counter.update(4, "Car", (5, 15)) == [("Line_2", IN)]
    assert counter.update(4, "Car", (5, -5)) == []  # back across both lines: deduplicated


def test_lines_json_round_trip() -> None:
    lines = [CountingLine("Door", (1, 2), (3, 4)), CountingLine("Gate", (5, 6), (7, 8))]
    data = lines_to_data(lines)
    assert data == [{"name": "Door", "points": [[1, 2], [3, 4]]},
                    {"name": "Gate", "points": [[5, 6], [7, 8]]}]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lines.json")
        save_lines(lines, path)
        with open(path, encoding="utf-8") as fh:
            assert json.load(fh) == data
        loaded = load_lines(path)
    assert [(ln.name, ln.p1, ln.p2) for ln in loaded] == [("Door", (1, 2), (3, 4)),
                                                          ("Gate", (5, 6), (7, 8))]


def test_lines_from_data_defaults_and_validation() -> None:
    assert lines_from_data([{"points": [[0, 0], [1, 1]]}])[0].name == "Line_1"
    try:
        lines_from_data([{"points": [[0, 0]]}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a one-point line")


def test_step_starting_on_the_line_uses_the_side_it_moves_to() -> None:
    # negative -> exactly on the line -> positive is a net IN movement.
    counter = LineCounter([_horizontal()])
    counter.update(1, "Person", (5, -5))
    assert counter.update(1, "Person", (5, 0)) == []        # touching only: not yet counted
    assert counter.update(1, "Person", (5, 5)) == [("Line_1", IN)]
    # positive -> exactly on the line is counted as OUT when it reaches the line.
    counter.update(2, "Person", (6, 5))
    assert counter.update(2, "Person", (6, 0)) == [("Line_1", OUT)]
    assert counter.update(2, "Person", (6, -5)) == []       # already counted


def test_lines_from_data_rejects_malformed_input() -> None:
    bad_inputs = [
        {"points": [[0, 0], [1, 1]]},          # top level is not a list
        [[[0, 0], [1, 1]]],                    # entry is not an object
        [{"points": [[0, 0], [1, "a"]]}],      # non-numeric coordinate
        [{"points": [[0, 0], [1, 1], [2, 2]]}],  # three points
    ]
    for data in bad_inputs:
        try:
            lines_from_data(data)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {data!r}")


def test_load_lines_accepts_utf8_bom_and_save_creates_folders() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bom_path = os.path.join(tmp, "bom.json")
        with open(bom_path, "w", encoding="utf-8-sig") as fh:
            json.dump([{"name": "A", "points": [[0, 0], [5, 5]]}], fh)
        assert load_lines(bom_path)[0].p2 == (5, 5)
        nested = os.path.join(tmp, "sub", "dir", "lines.json")
        save_lines([_horizontal()], nested)
        assert load_lines(nested)[0].name == "Line_1"


def _cooldown_line(frames: int = 15, px: float = 40.0) -> CountingLine:
    return CountingLine("L", (0, 0), (100, 0), cooldown_frames=frames, cooldown_px=px)


def test_cooldown_off_by_default_counts_both_ids() -> None:
    line = _horizontal()
    assert line.update(1, "Person", (5, -5), (5, 5), frame=10) == IN
    assert line.update(2, "Person", (6, -5), (6, 5), frame=11) == IN
    assert line.counts["Person"] == {IN: 2, OUT: 0}


def test_cooldown_suppresses_nearby_repeat_same_direction() -> None:
    line = _cooldown_line()
    assert line.update(1, "Person", (50, -5), (50, 5), frame=10) == IN
    assert line.update(2, "Person", (60, -5), (60, 5), frame=14) is None
    assert line.counts["Person"] == {IN: 1, OUT: 0}
    assert 2 in line.counted_ids  # the ghost ID cannot be counted later either


def test_cooldown_keeps_far_away_crossing() -> None:
    line = _cooldown_line(px=40.0)
    assert line.update(1, "Person", (10, -5), (10, 5), frame=10) == IN
    assert line.update(2, "Person", (90, -5), (90, 5), frame=11) == IN


def test_cooldown_keeps_late_crossing() -> None:
    line = _cooldown_line(frames=15)
    assert line.update(1, "Person", (50, -5), (50, 5), frame=10) == IN
    assert line.update(2, "Person", (52, -5), (52, 5), frame=26) == IN


def test_cooldown_keeps_opposite_direction() -> None:
    line = _cooldown_line()
    assert line.update(1, "Person", (50, -5), (50, 5), frame=10) == IN
    assert line.update(2, "Person", (52, 5), (52, -5), frame=11) == OUT


def test_line_counter_passes_frame_to_cooldown() -> None:
    line = _cooldown_line()
    counter = LineCounter([line], ["Person"])
    counter.update(1, "Person", (50, -5), frame=1)
    assert counter.update(1, "Person", (50, 5), frame=2) == [("L", IN)]
    counter.update(2, "Person", (55, -5), frame=3)
    assert counter.update(2, "Person", (55, 5), frame=4) == []
    assert line.counts["Person"] == {IN: 1, OUT: 0}


def test_max_gap_skips_the_step_after_a_long_absence() -> None:
    line = _horizontal()
    counter = LineCounter([line], ["Person"], max_gap_frames=30)
    counter.update(1, "Person", (5, -5), frame=10)
    assert counter.update(1, "Person", (5, 5), frame=100) == []  # 90 frames old: not tested
    assert line.counts["Person"] == {IN: 0, OUT: 0}
    assert counter.previous_positions[1] == (5, 5)  # the anchor is refreshed


def test_max_gap_keeps_short_gaps_and_is_off_by_default() -> None:
    line = _horizontal()
    counter = LineCounter([line], ["Person"], max_gap_frames=30)
    counter.update(1, "Person", (5, -5), frame=10)
    assert counter.update(1, "Person", (5, 5), frame=40) == [("Line_1", IN)]  # exactly 30
    other = LineCounter([_horizontal()], ["Person"])
    other.update(1, "Person", (5, -5), frame=10)
    assert other.update(1, "Person", (5, 5), frame=100000) == [("Line_1", IN)]


# --- crossing hysteresis (min_cross_distance) -------------------------------

def _hyst(min_dist: float = 10.0) -> tuple:
    """Return a horizontal line (positive side = y > 0) and a counter with a 10 % band
    for boxes 100 px high."""
    line = CountingLine("L", (0, 0), (100, 0))
    return line, LineCounter([line], ["Person"], min_cross_distance=min_dist / 100.0)


def _walk(counter: LineCounter, track_id: int, ys, x: int = 50) -> list:
    events = []
    for frame, y in enumerate(ys, start=1):
        events += counter.update(track_id, "Person", (x, y), frame, 100.0)
    return events


def test_hysteresis_ignores_jitter_around_the_line() -> None:
    line, counter = _hyst()
    assert _walk(counter, 1, [-3, 4, -2, 5, -4, 3, -1, 6]) == []
    assert line.counts["Person"] == {IN: 0, OUT: 0}


def test_hysteresis_counts_a_clean_crossing() -> None:
    line, counter = _hyst()
    assert _walk(counter, 1, [-30, -15, 2, 20]) == [("L", IN)]
    assert line.counts["Person"] == {IN: 1, OUT: 0}


def test_hysteresis_counts_out_direction() -> None:
    _, counter = _hyst()
    assert _walk(counter, 1, [30, 12, -3, -15]) == [("L", OUT)]


def test_hysteresis_counts_a_slow_walker_when_the_distance_is_reached() -> None:
    _, counter = _hyst()
    ys = [-30, -12, -3, 1, 3, 5, 7, 9, 9, 11]  # needs many frames inside the band
    assert _walk(counter, 1, ys[:-1]) == []
    assert counter.update(1, "Person", (50, ys[-1]), 99, 100.0) == [("L", IN)]


def test_hysteresis_needs_the_old_side_to_be_far_enough() -> None:
    _, counter = _hyst()
    # first seen inside the band, then moves away: no earlier confirmed side
    assert _walk(counter, 1, [2, 4, 8, 20]) == []


def test_hysteresis_jitter_over_the_line_then_return_is_not_a_crossing() -> None:
    _, counter = _hyst()
    assert _walk(counter, 1, [-30, -5, 4, 8, -6, -30]) == []


def test_hysteresis_still_counts_a_later_real_crossing_after_jitter() -> None:
    _, counter = _hyst()
    assert _walk(counter, 1, [-30, -5, 6, -8, -30]) == []
    assert _walk(counter, 1, [-30, 40]) == [("L", IN)]  # the same track really crosses later


def test_hysteresis_does_not_count_a_walk_around_the_segment_end() -> None:
    _, counter = _hyst()
    # the infinite line is crossed, the drawn segment (x 0..100) is not
    assert _walk(counter, 1, [-30, -12, 12, 30], x=150) == []


def test_hysteresis_scales_with_the_box_height() -> None:
    line = CountingLine("L", (0, 0), (100, 0))
    counter = LineCounter([line], ["Person"], min_cross_distance=0.1)
    # 20 px beyond the line is >= 10 % of a 100 px box, but < 10 % of a 300 px box
    counter.update(1, "Person", (50, -30), 1, 100.0)
    assert counter.update(1, "Person", (50, 20), 2, 100.0) == [("L", IN)]
    counter.update(2, "Person", (50, -30), 1, 300.0)
    assert counter.update(2, "Person", (50, 20), 2, 300.0) == []


def test_hysteresis_off_keeps_the_original_behaviour() -> None:
    line = CountingLine("L", (0, 0), (100, 0))
    counter = LineCounter([line], ["Person"])
    counter.update(1, "Person", (50, -1), 1, 100.0)
    assert counter.update(1, "Person", (50, 1), 2, 100.0) == [("L", IN)]


def test_hysteresis_does_not_count_a_jump_after_a_long_gap() -> None:
    line = CountingLine("L", (0, 0), (100, 0))
    counter = LineCounter([line], ["Person"], max_gap_frames=10, min_cross_distance=0.1)
    counter.update(1, "Person", (50, -30), 1, 100.0)
    assert counter.update(1, "Person", (50, 30), 100, 100.0) == []  # stale anchor: no hit


def test_suppressed_crossing_is_consumed_but_not_counted() -> None:
    line = CountingLine("L", (0, 0), (100, 0))
    assert line.update(1, "Person", (50, -5), (50, 5), suppress=True) is None
    assert 1 in line.counted_ids
    assert line.counts == {}


# --- riders ------------------------------------------------------------------

def test_riding_ids_flags_a_person_on_a_two_wheeler() -> None:
    tracks = [(1, 0, (100, 100, 140, 180)), (2, 3, (90, 140, 170, 200)), (3, 0, (400, 100, 440, 180))]
    assert riding_ids(tracks, 0, (1, 3), 0.3) == {1}


def test_riding_ids_needs_enough_overlap() -> None:
    tracks = [(1, 0, (100, 100, 140, 180)), (2, 1, (130, 170, 200, 230))]
    assert riding_ids(tracks, 0, (1, 3), 0.5) == set()
    assert riding_ids(tracks, 0, (1, 3), 0.01) == {1}


def test_riding_ids_ignores_cars_and_is_off_at_zero() -> None:
    tracks = [(1, 0, (100, 100, 140, 180)), (2, 2, (90, 140, 170, 200)), (3, 3, (90, 140, 170, 200))]
    assert riding_ids(tracks[:2], 0, (1, 3), 0.3) == set()
    assert riding_ids(tracks, 0, (1, 3), 0.0) == set()


def test_line_counter_suppress_flag_skips_the_count() -> None:
    line = CountingLine("L", (0, 0), (100, 0))
    counter = LineCounter([line], ["Person"])
    counter.update(1, "Person", (50, -5), 1)
    assert counter.update(1, "Person", (50, 5), 2, suppress=True) == []
    assert line.counts["Person"] == {IN: 0, OUT: 0}
