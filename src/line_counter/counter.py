"""Per-line, per-class, per-direction crossing counter (standard library only)."""

from __future__ import annotations

import math
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
        cooldown_frames: If > 0, a crossing is not counted when another one in
            the same direction was counted on this line at most this many
            frames earlier and within ``cooldown_px`` pixels (opt-in, 0 = off).
            Meant to absorb ID switches that re-create the same crossing.
        cooldown_px: Radius in pixels used together with ``cooldown_frames``.
        recent: Recent counted crossings as ``(frame, point, direction)``.
        sides: Per-track hysteresis state ``{track_id: [side, touched]}``: the
            last side (+1/-1) the track was confirmed on, i.e. at least the
            minimum distance away from the line, and whether the track's path
            touched the segment since then (only used with ``min_dist``).
    """

    name: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    counted_ids: Set[int] = field(default_factory=set)
    cooldown_frames: int = 0
    cooldown_px: float = 0.0
    recent: List[Tuple[int, Point, str]] = field(default_factory=list)
    sides: Dict[int, List] = field(default_factory=dict)

    def _is_duplicate(self, frame: int, point: Point, direction: str) -> bool:
        """Tell whether a crossing repeats a recent one (cooldown)."""
        if self.cooldown_frames <= 0:
            return False
        self.recent = [r for r in self.recent if frame - r[0] <= self.cooldown_frames]
        return any(
            d == direction and math.hypot(point[0] - pt[0], point[1] - pt[1]) <= self.cooldown_px
            for _, pt, d in self.recent
        )

    def update(self, track_id: int, cls_name: str, prev: Optional[Point], curr: Point,
               frame: int = 0, min_dist: float = 0.0, suppress: bool = False) -> Optional[str]:
        """Check one trajectory step against this line and count it.

        Without ``min_dist`` a crossing is counted when the segment
        ``prev -> curr`` intersects the line and the track has not been counted
        on this line before. The direction is IN when the point ends on the
        positive side of ``p1 -> p2`` (it came from the negative side or from
        the line itself) and OUT otherwise. For any step that strictly crosses
        the line this is the same as "negative -> positive is IN"; it also
        keeps a step that starts exactly on the line consistent with the side
        it moves to.

        With ``min_dist`` > 0 (hysteresis) a crossing is counted only when the
        point is at least ``min_dist`` pixels (perpendicular to the line) on
        the new side and was at least that far on the other side before, with
        the path touching the segment in between. A point jittering inside the
        band around the line therefore never counts, while a slow walker is
        counted as soon as the distance is reached, however many frames later.

        Args:
            track_id: Tracker ID of the object.
            cls_name: Class name used as the counting bucket.
            prev: Previous bottom-center point, or None when the step must not
                be tested (first sighting, or a stale anchor after a long gap).
            curr: Current bottom-center point of the object.
            frame: Source frame index, only used by the optional cooldown.
            min_dist: Hysteresis distance in pixels (0 = off).
            suppress: Consume a crossing found in this step without counting
                it (a person riding a bicycle or motorcycle). The track ID is
                still marked as counted.

        Returns:
            ``"IN"`` or ``"OUT"`` if a new crossing was counted, else None.
            A crossing suppressed by the cooldown or ``suppress`` returns None,
            but its track ID is still marked as counted.
        """
        if track_id in self.counted_ids:
            return None
        if min_dist > 0:
            direction = self._hysteresis_step(track_id, prev, curr, min_dist)
        elif prev is not None and segments_intersect(self.p1, self.p2, prev, curr):
            # segments_intersect() only accepts steps where exactly one endpoint
            # is strictly on the positive side, so the sign of curr decides.
            direction = IN if side_of_line(self.p1, self.p2, curr) > 0 else OUT
        else:
            direction = None
        if direction is None:
            return None
        self.counted_ids.add(track_id)
        if suppress or self._is_duplicate(frame, curr, direction):
            return None
        if self.cooldown_frames > 0:
            self.recent.append((frame, curr, direction))
        bucket = self.counts.setdefault(cls_name, {IN: 0, OUT: 0})
        bucket[direction] += 1
        return direction

    def _hysteresis_step(self, track_id: int, prev: Optional[Point], curr: Point,
                         min_dist: float) -> Optional[str]:
        """Advance the per-track side state and return a confirmed crossing."""
        state = self.sides.get(track_id)
        if state is not None and prev is not None and segments_intersect(self.p1, self.p2, prev, curr):
            state[1] = True
        length = math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])
        if length == 0:
            return None
        dist = side_of_line(self.p1, self.p2, curr) / length
        if abs(dist) < min_dist:
            return None  # inside the band: undecided, keep the old side
        side = 1 if dist > 0 else -1
        if state is None:
            self.sides[track_id] = [side, False]
            return None
        crossed = state[0] != side and state[1]
        state[0], state[1] = side, False
        return (IN if side > 0 else OUT) if crossed else None


class LineCounter:
    """Keeps the last position of each track and updates every line.

    Args:
        lines: Counting lines to update.
        class_names: Class names to pre-create zeroed buckets for.
        max_gap_frames: If > 0, a step whose previous position is more than
            this many source frames old is not tested for a crossing (a track
            that vanished and came back with the same ID). 0 = off.
        min_cross_distance: Hysteresis as a fraction of the box height (0 = off);
            see :meth:`CountingLine.update`. Needs ``box_height`` in :meth:`update`.
    """

    def __init__(self, lines: Iterable[CountingLine], class_names: Iterable[str] = (),
                 max_gap_frames: int = 0, min_cross_distance: float = 0.0) -> None:
        self.lines: List[CountingLine] = list(lines)
        self.max_gap_frames = max_gap_frames
        self.min_cross_distance = min_cross_distance
        self.previous_positions: Dict[int, Point] = {}
        self.previous_frames: Dict[int, int] = {}
        names = list(class_names)
        for line in self.lines:
            for name in names:
                line.counts.setdefault(name, {IN: 0, OUT: 0})

    def update(self, track_id: int, cls_name: str, curr: Point,
               frame: int = 0, box_height: float = 0.0,
               suppress: bool = False) -> List[Tuple[str, str]]:
        """Register a new position for a track and count any crossings.

        Args:
            track_id: Tracker ID of the object.
            cls_name: Class name of the object in the current frame.
            curr: Current bottom-center point.
            frame: Source frame index (used by the optional per-line cooldown).
            box_height: Height of the object's box in pixels; the hysteresis
                distance is ``min_cross_distance * box_height``.
            suppress: Do not count a crossing found in this step (rider).

        Returns:
            A list of ``(line_name, direction)`` for crossings counted now.
        """
        events: List[Tuple[str, str]] = []
        prev = self.previous_positions.get(track_id)
        if prev is not None and self.max_gap_frames > 0:
            if frame - self.previous_frames.get(track_id, frame) > self.max_gap_frames:
                prev = None  # stale anchor: the jump is not a real movement
        min_dist = self.min_cross_distance * box_height
        if prev is not None or min_dist > 0:  # hysteresis also needs the first sighting
            for line in self.lines:
                direction = line.update(track_id, cls_name, prev, curr, frame, min_dist, suppress)
                if direction is not None:
                    events.append((line.name, direction))
        self.previous_positions[track_id] = curr
        self.previous_frames[track_id] = frame
        return events

    def summary(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Return a copy of all counts as ``{line: {class: {"IN", "OUT"}}}``."""
        return {
            line.name: {cls: dict(dirs) for cls, dirs in line.counts.items()}
            for line in self.lines
        }


def riding_ids(tracks: Iterable[Tuple[int, int, Tuple[float, float, float, float]]],
               person_class: int, vehicle_classes: Iterable[int], overlap: float) -> Set[int]:
    """Return the IDs of persons whose box overlaps a bicycle or motorcycle box.

    Args:
        tracks: ``(track_id, class_id, (x1, y1, x2, y2))`` of one frame.
        person_class: Class ID of persons.
        vehicle_classes: Class IDs of the two-wheelers (bicycle, motorcycle).
        overlap: Minimum intersection area as a fraction of the person box
            area; 0 or less returns an empty set (feature off).

    Returns:
        The track IDs of persons considered to be riding.
    """
    if overlap <= 0:
        return set()
    vehicles = {int(c) for c in vehicle_classes}
    items = list(tracks)
    boxes = [box for _, cls, box in items if cls in vehicles]
    riders: Set[int] = set()
    for track_id, cls, (x1, y1, x2, y2) in items:
        area = (x2 - x1) * (y2 - y1)
        if cls != person_class or area <= 0:
            continue
        for vx1, vy1, vx2, vy2 in boxes:
            iw = min(x2, vx2) - max(x1, vx1)
            ih = min(y2, vy2) - max(y1, vy1)
            if iw > 0 and ih > 0 and iw * ih / area >= overlap:
                riders.add(track_id)
                break
    return riders
