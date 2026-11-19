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
