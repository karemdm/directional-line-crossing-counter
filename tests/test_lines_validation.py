"""Validation of the lines file and zero/edge geometry (standard library only)."""

import json
import logging

import pytest

from line_counter.counter import LineCounter
from line_counter.lines_io import lines_from_data, load_lines


@pytest.mark.parametrize("data", [
    [{"points": [[0, 0], [float("inf"), 1]]}],      # JSON "Infinity" parses to inf; int() overflows
    [{"points": [[0, 0], [float("nan"), 1]]}],
    [{"points": [[0, 0], [None, 1]]}],
    [{"points": "ab"}],
    [{"name": "A", "points": [[0, 0], [1, 1]]}, {"name": "A", "points": [[2, 2], [3, 3]]}],  # same name
    [{"name": "", "points": [[0, 0], [1, 1]]}],
    [None],
    "not a list",
    None,
])
def test_bad_lines_are_rejected_with_value_error(data):
    with pytest.raises(ValueError):
        lines_from_data(data)


def test_empty_list_gives_no_lines():
    assert lines_from_data([]) == []


def test_float_coordinates_are_rounded_not_truncated():
    (line,) = lines_from_data([{"points": [[1.6, 2.4], [10.5, 3.5]]}])
    assert line.p1 == (2, 2)
    assert line.p2 == (10, 4)  # round half to even


def test_load_lines_reports_invalid_json_as_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_lines(path)


def test_zero_length_line_warns_and_never_counts(caplog):
    with caplog.at_level(logging.WARNING):
        (line,) = lines_from_data([{"points": [[5, 5], [5, 5]]}])
    assert "never be crossed" in caplog.text
    counter = LineCounter([line])
    counter.update(1, "p", (0, 0))
    assert counter.update(1, "p", (10, 10)) == []
    assert counter.update(1, "p", (5, 5)) == []


def test_shipped_sample_lines_load():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "samples"
    for name, n in (("abbey_road_lines.json", 4), ("shinjuku_lines.json", 12)):
        lines = load_lines(root / name)
        assert len(lines) == n
        assert len({ln.name for ln in lines}) == n
        assert all(0 <= x <= 1920 and 0 <= y <= 1080 for ln in lines for x, y in (ln.p1, ln.p2))
        json.dumps([ln.name for ln in lines])
