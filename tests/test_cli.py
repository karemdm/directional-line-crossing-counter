"""Tests for the CLI options (standard library only; no model is loaded)."""

import pytest

from line_counter.cli import build_parser, config_from_args


def _config(*argv: str):
    return config_from_args(build_parser().parse_args(list(argv)))


def test_defaults_keep_new_options_off() -> None:
    cfg = _config()
    assert cfg.frame_step == 1
    assert cfg.cooldown_frames == 0


def test_new_options_reach_the_config() -> None:
    cfg = _config("--frame-step", "3", "--cooldown-frames", "20", "--cooldown-px", "55")
    assert (cfg.frame_step, cfg.cooldown_frames, cfg.cooldown_px) == (3, 20, 55.0)


@pytest.mark.parametrize("argv", [["--frame-step", "0"], ["--cooldown-frames", "-1"]])
def test_invalid_values_are_rejected(argv) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_numeric_options_are_range_checked() -> None:
    for argv in (["--conf", "1.5"], ["--conf", "-0.1"], ["--imgsz", "0"], ["--cooldown-px", "-3"],
                 ["--conf", "nan"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)
    assert _config("--conf", "0", "--imgsz", "32").imgsz == 32


def _main(*argv: str) -> int:
    from line_counter.cli import main

    return main(list(argv))


def test_missing_input_file_exits_with_error(tmp_path, caplog) -> None:
    lines = tmp_path / "l.json"
    lines.write_text('[{"points": [[0, 0], [5, 5]]}]', encoding="utf-8")
    assert _main("--input", str(tmp_path / "nope.mp4"), "--lines", str(lines), "--no-display") == 1
    assert "nope.mp4" in caplog.text


def test_output_equal_to_input_is_refused(tmp_path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    lines = tmp_path / "l.json"
    lines.write_text('[{"points": [[0, 0], [5, 5]]}]', encoding="utf-8")
    assert _main("--input", str(video), "--output", str(video), "--lines", str(lines), "--no-display") == 1
    assert video.read_bytes() == b"x"


def test_save_and_replay_tracks_conflict_is_reported_before_anything_else(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _main("--input", str(tmp_path / "v.mp4"), "--no-display", "--lines", str(tmp_path / "l.json"),
              "--save-tracks", "a.jsonl", "--replay-tracks", "b.jsonl")


def test_processing_errors_become_exit_code_1(tmp_path, monkeypatch, caplog) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"not a video")
    lines = tmp_path / "l.json"
    lines.write_text('[{"points": [[0, 0], [5, 5]]}]', encoding="utf-8")
    tracks = tmp_path / "t.jsonl"
    tracks.write_text("", encoding="utf-8")
    code = _main("--input", str(video), "--lines", str(lines), "--no-display",
                 "--replay-tracks", str(tracks), "--output", str(tmp_path / "o.mp4"))
    assert code == 1
    assert "Cannot open video" in caplog.text or "No frames" in caplog.text


def test_invalid_tracks_file_becomes_exit_code_1(tmp_path, caplog) -> None:
    import cv2
    import numpy as np

    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    w.write(np.zeros((32, 32, 3), np.uint8))
    w.release()
    lines = tmp_path / "l.json"
    lines.write_text('[{"points": [[0, 0], [5, 5]]}]', encoding="utf-8")
    tracks = tmp_path / "t.jsonl"
    tracks.write_text("garbage\n", encoding="utf-8")
    assert _main("--input", str(video), "--lines", str(lines), "--no-display",
                 "--replay-tracks", str(tracks), "--output", str(tmp_path / "o.mp4")) == 1


def test_warns_when_options_are_silently_useless(tmp_path, caplog) -> None:
    from line_counter.cli import describe_ignored_options

    cfg = _config("--replay-tracks", "t.jsonl", "--blur-faces")
    assert any("--blur-faces" in m for m in describe_ignored_options(cfg))
    cfg = _config("--blur-faces", "--classes", "2", "5")
    assert any("person" in m for m in describe_ignored_options(cfg))
    assert describe_ignored_options(_config("--blur-faces")) == []
    assert describe_ignored_options(_config()) == []


def test_max_track_gap_and_blur_padding_options() -> None:
    cfg = _config()
    assert cfg.max_track_gap == 0.0 and cfg.blur_padding == 0.15
    cfg = _config("--max-track-gap", "2", "--blur-padding", "0.3")
    assert (cfg.max_track_gap, cfg.blur_padding) == (2.0, 0.3)
    for argv in (["--max-track-gap", "-1"], ["--blur-padding", "-0.1"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)


def test_min_cross_distance_and_rider_overlap_options() -> None:
    cfg = _config()
    assert cfg.min_cross_distance == 0.0 and cfg.rider_overlap == 0.0
    cfg = _config("--min-cross-distance", "0.1", "--rider-overlap", "0.5")
    assert (cfg.min_cross_distance, cfg.rider_overlap) == (0.1, 0.5)
    for argv in (["--min-cross-distance", "-0.1"], ["--rider-overlap", "1.5"], ["--rider-overlap", "-1"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)


def test_rider_overlap_warns_without_a_two_wheeler_class() -> None:
    from line_counter.cli import describe_ignored_options

    # the default classes include 3 (motorcycle): no warning; without 1 and 3 there is one
    assert not describe_ignored_options(_config("--rider-overlap", "0.5"))
    assert describe_ignored_options(_config("--rider-overlap", "0.5", "--classes", "0", "2"))
    assert not describe_ignored_options(_config("--classes", "0", "2"))
