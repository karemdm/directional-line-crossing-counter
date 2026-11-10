"""Load and save counting lines as JSON (standard library only).

File format::

    [
      {"name": "Line_1", "points": [[x1, y1], [x2, y2]]}
    ]

Coordinates are pixels in the original video resolution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Union

from line_counter.counter import CountingLine

logger = logging.getLogger(__name__)

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
        ValueError: If the data is not a list of objects, or an entry does not
            have exactly two numeric ``[x, y]`` points.
    """
    if not isinstance(data, list):
        raise ValueError("Lines file must contain a JSON list of line objects")
    lines: List[CountingLine] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Line entry {i} must be an object with a 'points' key")
        points = item.get("points", [])
        if (not isinstance(points, (list, tuple)) or len(points) != 2
                or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in points)):
            raise ValueError(f"Line entry {i} must have exactly two [x, y] points")
        try:
            (x1, y1), (x2, y2) = ((int(x), int(y)) for x, y in points)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Line entry {i} has non-numeric coordinates") from exc
        name = str(item.get("name", f"Line_{i}"))
        if (x1, y1) == (x2, y2):
            logger.warning("%s has two identical points; it can never be crossed", name)
        lines.append(CountingLine(name, (x1, y1), (x2, y2)))
    return lines


def load_lines(path: PathLike) -> List[CountingLine]:
    """Read counting lines from a JSON file.

    Args:
        path: JSON file path.

    Returns:
        The counting lines.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:  # tolerate a UTF-8 BOM (Windows editors)
        return lines_from_data(json.load(fh))


def save_lines(lines: List[CountingLine], path: PathLike) -> None:
    """Write counting lines to a JSON file.

    Args:
        lines: Counting lines.
        path: Destination JSON file path (parent folders are created).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(lines_to_data(lines), fh, indent=2)
