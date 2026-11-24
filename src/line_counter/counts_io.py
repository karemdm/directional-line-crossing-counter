"""Final counts report as JSON (standard library only).

File format (``--save-counts``)::

    {
      "input": "video.mp4",
      "mode": "detect",                  # or "replay" with --replay-tracks
      "settings": {"model": "...", "conf": 0.15, ...},
      "lines": [{"name": "Line_1", "points": [[x1, y1], [x2, y2]]}],
      "counts": {"Line_1": {"person": {"IN": 3, "OUT": 5}}}
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

from line_counter.config import ProcessingConfig
from line_counter.counter import CountingLine
from line_counter.lines_io import lines_to_data

Counts = Dict[str, Dict[str, Dict[str, int]]]


def build_report(input_path: Union[str, Path], lines: List[CountingLine],
                 counts: Counts, config: ProcessingConfig) -> dict:
    """Build the JSON-serialisable counts report of a finished run.

    Args:
        input_path: Source video (only the file name is stored).
        lines: Counting lines (names and points).
        counts: Final ``{line: {class: {"IN", "OUT"}}}`` counts.
        config: Settings of the run. In replay mode only the tracks file
            name and the cooldown are stored (no detector ran and the frame
            step comes from the tracks file).

    Returns:
        The report dictionary.
    """
    settings: dict
    if config.replay_tracks:
        mode = "replay"
        settings = {"replay_tracks": Path(config.replay_tracks).name}
    else:
        mode = "detect"
        settings = {"model": Path(config.model).name, "conf": config.conf, "imgsz": config.imgsz,
                    "classes": list(config.classes), "tracker": Path(config.tracker).name,
                    "frame_step": config.frame_step}
    settings["cooldown_frames"] = config.cooldown_frames
    settings["cooldown_px"] = config.cooldown_px
    if config.max_track_gap > 0:
        settings["max_track_gap"] = config.max_track_gap
    return {"input": Path(input_path).name, "mode": mode, "settings": settings,
            "lines": lines_to_data(lines), "counts": counts}


def save_report(path: Union[str, Path], report: dict) -> None:
    """Write a report as indented JSON, creating the parent folder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
