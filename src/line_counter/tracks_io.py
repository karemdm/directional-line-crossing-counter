"""Save and load per-frame tracks as JSON Lines (standard library only)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, IO, List, Sequence, Union


def track_record(track_id: int, cls_id: int, cls_name: str, conf: float,
                 box: Sequence[int]) -> dict:
    """Build the JSON-serializable record of one tracked object."""
    return {"id": int(track_id), "cls": int(cls_id), "name": cls_name,
            "conf": round(float(conf), 4), "box": [int(v) for v in box]}


def write_frame_tracks(fh: IO[str], frame_idx: int, tracks: List[dict]) -> None:
    """Write one line ``{"frame": n, "tracks": [...]}`` and flush it."""
    fh.write(json.dumps({"frame": int(frame_idx), "tracks": tracks}, separators=(",", ":")) + "\n")
    fh.flush()  # a stopped run still leaves a usable file


def _check_track(t: dict) -> None:
    """Raise if a track record lacks ``id``, ``cls``, ``name`` or a 4-number ``box``."""
    int(t["id"]), int(t["cls"]), t["name"]
    box = t["box"]
    if len(box) != 4:
        raise ValueError("box must have 4 values")
    for v in box:
        int(v)


def load_tracks(path: Union[str, Path]) -> Dict[int, List[dict]]:
    """Load a tracks file into ``{frame_index: [track records]}``.

    Raises:
        ValueError: If a line is not a valid frame record (missing or
            malformed ``id``/``cls``/``name``/``box``) or a frame appears twice.
    """
    out: Dict[int, List[dict]] = {}
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
                frame = int(rec["frame"])
                tracks = list(rec["tracks"])
                for t in tracks:
                    _check_track(t)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{n}: invalid track record ({exc!r})") from exc
            if frame in out:
                raise ValueError(f"{path}:{n}: duplicate frame {frame}")
            out[frame] = tracks
    return out


def infer_frame_step(frames: Sequence[int], default: int = 1) -> int:
    """Infer the frame step used when saving (gcd of ``frame - 1``, 1-based indices)."""
    g = 0
    for f in frames:
        g = math.gcd(g, int(f) - 1)
    return g if g > 0 else default
