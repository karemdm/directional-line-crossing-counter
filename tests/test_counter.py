"""Tests for line_counter.counter and line_counter.lines_io (standard library only)."""

import json
import os
import tempfile

from line_counter.counter import IN, OUT, CountingLine, LineCounter
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
