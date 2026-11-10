"""Per-line, per-class, per-direction crossing counter (standard library only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from line_counter.geometry import Point, segments_intersect, side_of_line

IN = "IN"
OUT = "OUT"


@dataclass
class CountingLine:
    """A user-drawn counting segment and its counts.

    Attributes:
        name: Display name, e.g. ``"Line_1"``.
        p1: First endpoint in original video pixel coordinates.
        p2: Second endpoint in original video pixel coordinates.
        counts: ``{class_name: {"IN": n, "OUT": n}}``.
        counted_ids: Track IDs already counted on this line (any class).
    """

    name: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    counted_ids: Set[int] = field(default_factory=set)

    def update(self, track_id: int, cls_name: str, prev: Point, curr: Point) -> Optional[str]:
        """Check one trajectory step against this line and count it.

        A crossing is counted when the segment ``prev -> curr`` intersects the
        line and the track has not been counted on this line before. The
        direction is IN when the point ends on the positive side of
        ``p1 -> p2`` (it came from the negative side or from the line itself)
        and OUT otherwise. For any step that strictly crosses the line this is
        the same as "negative -> positive is IN"; it also keeps a step that
        starts exactly on the line consistent with the side it moves to.

        Args:
            track_id: Tracker ID of the object.
            cls_name: Class name used as the counting bucket.
            prev: Previous bottom-center point of the object.
            curr: Current bottom-center point of the object.

        Returns:
            ``"IN"`` or ``"OUT"`` if a new crossing was counted, else None.
        """
        if track_id in self.counted_ids:
            return None
        if not segments_intersect(self.p1, self.p2, prev, curr):
            return None
        # segments_intersect() only accepts steps where exactly one endpoint is
        # strictly on the positive side, so the sign of curr decides.
        direction = IN if side_of_line(self.p1, self.p2, curr) > 0 else OUT
        bucket = self.counts.setdefault(cls_name, {IN: 0, OUT: 0})
        bucket[direction] += 1
        self.counted_ids.add(track_id)
        return direction


class LineCounter:
    """Keeps the last position of each track and updates every line.

    Args:
        lines: Counting lines to update.
        class_names: Class names to pre-create zeroed buckets for.
    """

    def __init__(self, lines: Iterable[CountingLine], class_names: Iterable[str] = ()) -> None:
        self.lines: List[CountingLine] = list(lines)
        self.previous_positions: Dict[int, Point] = {}
        names = list(class_names)
        for line in self.lines:
            for name in names:
                line.counts.setdefault(name, {IN: 0, OUT: 0})

    def update(self, track_id: int, cls_name: str, curr: Point) -> List[Tuple[str, str]]:
        """Register a new position for a track and count any crossings.

        Args:
            track_id: Tracker ID of the object.
            cls_name: Class name of the object in the current frame.
            curr: Current bottom-center point.

        Returns:
            A list of ``(line_name, direction)`` for crossings counted now.
        """
        events: List[Tuple[str, str]] = []
        prev = self.previous_positions.get(track_id)
        if prev is not None:
            for line in self.lines:
                direction = line.update(track_id, cls_name, prev, curr)
                if direction is not None:
                    events.append((line.name, direction))
        self.previous_positions[track_id] = curr
        return events

    def summary(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Return a copy of all counts as ``{line: {class: {"IN", "OUT"}}}``."""
        return {
            line.name: {cls: dict(dirs) for cls, dirs in line.counts.items()}
            for line in self.lines
        }
