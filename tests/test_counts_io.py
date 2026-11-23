"""Tests for the counts report and the --save-counts option (standard library only)."""

import json

from line_counter.cli import build_parser, config_from_args
from line_counter.config import ProcessingConfig
from line_counter.counter import CountingLine, LineCounter
from line_counter.counts_io import build_report, save_report


def _counter() -> LineCounter:
    line = CountingLine("Line_1", (0, 10), (20, 10))
    counter = LineCounter([line], ["person", "car"])
    counter.update(1, "person", (10, 20))
    counter.update(1, "person", (10, 0))
    return counter


def test_report_has_counts_lines_and_settings() -> None:
    counter = _counter()
    cfg = ProcessingConfig(frame_step=3, conf=0.15)
    report = build_report("dir/clip.mp4", counter.lines, counter.summary(), cfg)
    assert report["input"] == "clip.mp4"
    assert report["mode"] == "detect"
    assert report["lines"] == [{"name": "Line_1", "points": [[0, 10], [20, 10]]}]
    assert report["settings"]["frame_step"] == 3 and report["settings"]["conf"] == 0.15
    assert sum(report["counts"]["Line_1"]["person"].values()) == 1
    assert report["counts"]["Line_1"]["car"] == {"IN": 0, "OUT": 0}


def test_replay_report_omits_detector_settings(tmp_path) -> None:
    counter = _counter()
    cfg = ProcessingConfig(replay_tracks=str(tmp_path / "t.jsonl"))
    report = build_report("clip.mp4", counter.lines, counter.summary(), cfg)
    assert report["mode"] == "replay"
    assert report["settings"]["replay_tracks"] == "t.jsonl"
    assert "model" not in report["settings"]


def test_save_report_round_trips(tmp_path) -> None:
    counter = _counter()
    report = build_report("clip.mp4", counter.lines, counter.summary(), ProcessingConfig())
    out = tmp_path / "sub" / "counts.json"
    save_report(out, report)
    assert json.loads(out.read_text(encoding="utf-8")) == report


def test_save_counts_option_is_parsed() -> None:
    args = build_parser().parse_args(["--save-counts", "c.json"])
    assert args.save_counts == "c.json"
    assert build_parser().parse_args([]).save_counts is None
    config_from_args(args)  # must not fail
