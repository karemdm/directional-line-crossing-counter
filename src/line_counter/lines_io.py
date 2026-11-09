"""Load and save counting lines as JSON (standard library only).

File format::

    [
      {"name": "Line_1", "points": [[x1, y1], [x2, y2]]}
    ]

Coordinates are pixels in the original video resolution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from line_counter.counter import CountingLine

PathLike = Union[str, Path]


def lines_to_data(lines: List[CountingLine]) -> list:
    """Convert lines to a JSON-serialisable list.

    Args:
        lines: Counting lines.

    Returns:
        A list of ``{"name", "points"}`` dictionaries.
    """
    return [{"name": ln.name, "points": [list(ln.p1), list(ln.p2)]} for ln in lines]


def lines_from_data(data: list) -> List[CountingLine]:
    """Build fresh (zero-count) lines from parsed JSON data.

    Args:
        data: A list of ``{"name", "points"}`` dictionaries. ``name`` is
            optional and defaults to ``Line_<n>``.

    Returns:
        The counting lines.

    Raises:
        ValueError: If an entry does not have exactly two 2D points.
    """
    lines: List[CountingLine] = []
    for i, item in enumerate(data, start=1):
        points = item.get("points", [])
        if len(points) != 2 or any(len(p) != 2 for p in points):
            raise ValueError(f"Line entry {i} must have exactly two [x, y] points")
        (x1, y1), (x2, y2) = points
        name = str(item.get("name", f"Line_{i}"))
        lines.append(CountingLine(name, (int(x1), int(y1)), (int(x2), int(y2))))
    return lines


def load_lines(path: PathLike) -> List[CountingLine]:
    """Read counting lines from a JSON file.

    Args:
        path: JSON file path.

    Returns:
        The counting lines.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return lines_from_data(json.load(fh))


def save_lines(lines: List[CountingLine], path: PathLike) -> None:
    """Write counting lines to a JSON file.

    Args:
        lines: Counting lines.
        path: Destination JSON file path.
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(lines_to_data(lines), fh, indent=2)
