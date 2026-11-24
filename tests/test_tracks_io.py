"""Tests for the tracks JSONL round trip and replay feeding."""

import numpy as np

from line_counter.counter import CountingLine, LineCounter
from line_counter.tracks_io import infer_frame_step, load_tracks, track_record, write_frame_tracks


def test_round_trip(tmp_path):
    path = tmp_path / "t.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        write_frame_tracks(fh, 1, [track_record(3, 0, "Person", 0.91234567, np.array([1, 2, 30, 40]))])
        write_frame_tracks(fh, 4, [])
    data = load_tracks(path)
    assert data[1] == [{"id": 3, "cls": 0, "name": "Person", "conf": 0.9123, "box": [1, 2, 30, 40]}]
    assert data[4] == []


def test_infer_step():
    assert infer_frame_step([1, 4, 7, 10]) == 3
    assert infer_frame_step([1]) == 1
    assert infer_frame_step([]) == 1


def test_replay_feeds_counter(tmp_path):
    path = tmp_path / "t.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for f, y in ((1, 80), (2, 120)):  # box bottom-center moves down across y=100
            write_frame_tracks(fh, f, [track_record(1, 0, "Person", 0.5, (40, y - 50, 60, y))])
    counter = LineCounter([CountingLine("L", (0, 100), (200, 100))], ["Person"])
    for f, tracks in sorted(load_tracks(path).items()):
        for t in tracks:
            x1, y1, x2, y2 = t["box"]
            counter.update(t["id"], t["name"], ((x1 + x2) // 2, y2), f)
    counts = counter.lines[0].counts["Person"]
    assert counts["IN"] + counts["OUT"] == 1


def test_load_tracks_rejects_incomplete_track_records(tmp_path):
    import pytest

    from line_counter.tracks_io import load_tracks

    bad_lines = [
        '{"frame": 1, "tracks": [{"id": 1, "cls": 0, "name": "p"}]}',                       # no box
        '{"frame": 1, "tracks": [{"id": 1, "cls": 0, "name": "p", "box": [1, 2, 3]}]}',     # 3 values
        '{"frame": 1, "tracks": [{"id": 1, "cls": 0, "name": "p", "box": [1, 2, "a", 4]}]}',
        '{"frame": 1, "tracks": [[1, 2]]}',                                                  # not an object
        '{"frame": 1, "tracks": 5}',
        '[1, 2]',
    ]
    for raw in bad_lines:
        path = tmp_path / "t.jsonl"
        path.write_text(raw + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid track record"):
            load_tracks(path)


def test_load_tracks_conf_is_optional_and_blank_lines_are_skipped(tmp_path):
    from line_counter.tracks_io import load_tracks

    path = tmp_path / "t.jsonl"
    path.write_text('\n{"frame": 2, "tracks": [{"id": 1, "cls": 0, "name": "p", "box": [1, 2, 3, 4]}]}\n\n',
                    encoding="utf-8")
    assert list(load_tracks(path)) == [2]


def test_load_tracks_reports_duplicate_frames(tmp_path):
    import pytest

    from line_counter.tracks_io import load_tracks

    path = tmp_path / "t.jsonl"
    rec = '{"frame": 1, "tracks": []}\n'
    path.write_text(rec + rec, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_tracks(path)


def test_infer_step_with_gaps_and_non_one_based_start():
    from line_counter.tracks_io import infer_frame_step

    assert infer_frame_step([1, 7, 13]) == 6
    assert infer_frame_step([4, 7, 10]) == 3  # a file that does not start at frame 1
    assert infer_frame_step([1, 4, 5]) == 1
