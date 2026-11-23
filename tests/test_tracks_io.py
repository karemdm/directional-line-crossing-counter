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
