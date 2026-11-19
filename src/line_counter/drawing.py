"""Frame annotations: corner-style boxes, counting lines and the counts HUD."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from line_counter.counter import IN, OUT, CountingLine

Color = Tuple[int, int, int]

BOX_COLOR: Color = (0, 255, 0)
POINT_COLOR: Color = (255, 0, 0)
LINE_COLOR: Color = (0, 0, 255)
TEXT_COLOR: Color = (255, 255, 255)
MUTED_COLOR: Color = (200, 200, 200)
IN_COLOR: Color = (0, 255, 0)
OUT_COLOR: Color = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_corner_box(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                    color: Color, label: str) -> None:
    """Draw a minimal corner-style bounding box with a label.

    Args:
        img: Frame to draw on (modified in place).
        pt1: Top-left corner ``(x1, y1)``.
        pt2: Bottom-right corner ``(x2, y2)``.
        color: BGR color.
        label: Text drawn above the box.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    length = max(5, int(min(x2 - x1, y2 - y1) * 0.15))
    t = 2
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, t)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, t)
    cv2.putText(img, label, (x1, max(y1 - 5, 12)), FONT, 0.4, color, 1)


def draw_track(img: np.ndarray, box: Sequence[int], track_id: int, cls_name: str,
               anchor: Tuple[int, int]) -> None:
    """Draw a tracked object: corner box, ``#id class`` label and anchor point.

    Args:
        img: Frame to draw on (modified in place).
        box: ``(x1, y1, x2, y2)`` in pixels.
        track_id: Tracker ID.
        cls_name: Class display name.
        anchor: Bottom-center point used for counting.
    """
    x1, y1, x2, y2 = box
    draw_corner_box(img, (x1, y1), (x2, y2), BOX_COLOR, f"#{track_id} {cls_name}")
    cv2.circle(img, anchor, 3, POINT_COLOR, -1)


def draw_counting_lines(img: np.ndarray, lines: Iterable[CountingLine]) -> None:
    """Draw every counting line with its name.

    Args:
        img: Frame to draw on (modified in place).
        lines: Counting lines in frame coordinates.
    """
    for line in lines:
        cv2.line(img, line.p1, line.p2, LINE_COLOR, 2)
        cv2.putText(img, line.name, (line.p1[0], line.p1[1] - 8), FONT, 0.6, LINE_COLOR, 2)


Segment = Tuple[str, Color]


def _text_width(text: str, scale: float, thickness: int) -> int:
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def _entry_segments(cls_name: str, data: Dict[str, int]) -> List[Segment]:
    return [
        (f"{cls_name}[", MUTED_COLOR),
        (f"{data[IN]} IN", IN_COLOR),
        (", ", MUTED_COLOR),
        (f"{data[OUT]} OUT", OUT_COLOR),
        ("]", MUTED_COLOR),
    ]


def draw_hud(img: np.ndarray, lines: Sequence[CountingLine], class_order: Iterable[str]) -> None:
    """Draw a translucent panel with per-line, per-class IN/OUT counts.

    Only classes with at least one crossing on a line are shown on that row,
    formatted as ``Line_1: Person[3 IN, 1 OUT]  Car[0 IN, 2 OUT]``. Text is
    measured with ``cv2.getTextSize`` and everything scales with the frame
    height (unchanged at 720p). The panel is as wide as its widest row; entries
    that do not fit the frame width wrap onto an indented continuation row.

    Args:
        img: Frame to draw on (modified in place).
        lines: Counting lines with their counts.
        class_order: Class names in display order.
    """
    frame_h, frame_w = img.shape[:2]
    order = list(class_order)
    k = max(frame_h / 720.0, 0.5)
    font_scale = 0.5 * k
    thick = max(1, round(k))
    margin = round(10 * k)
    pad = round(10 * k)
    row_h = round(22 * k)
    gap = _text_width("  ", font_scale, thick)
    max_row_w = max(frame_w - 2 * margin - 2 * pad, 1)

    # Layout: each row is (x_offset, [(text, color, x), ...]).
    rows: List[List[Tuple[str, Color, int]]] = []
    widest = 0
    for line in lines:
        title = f"{line.name}: "
        title_w = _text_width(title, font_scale, thick)
        indent = min(title_w, max_row_w // 2)
        row: List[Tuple[str, Color, int]] = [(title, TEXT_COLOR, 0)]
        x = title_w
        has_entry = False
        for cls_name in order:
            data = line.counts.get(cls_name)
            if not data or (data[IN] == 0 and data[OUT] == 0):
                continue
            segs = _entry_segments(cls_name, data)
            widths = [_text_width(t, font_scale, thick) for t, _ in segs]
            entry_w = sum(widths)
            if has_entry and x + entry_w > max_row_w:
                widest = max(widest, x)
                rows.append(row)
                row = []
                x = indent
                has_entry = False
            for (text, color), w in zip(segs, widths):
                row.append((text, color, x))
                x += w
            x += gap
            has_entry = True
        widest = max(widest, x - (gap if has_entry else 0))
        rows.append(row)

    panel_w = min(widest + 2 * pad, frame_w - 2 * margin)
    panel_h = round(20 * k) + len(rows) * row_h
    y0, x0 = margin, margin
    y1, x1 = min(y0 + panel_h, frame_h), min(x0 + panel_w, frame_w)
    # Same result as blending a black rectangle at 60% opacity, but only the
    # panel region is touched instead of copying the whole frame every time.
    panel = img[y0:y1, x0:x1]
    if panel.size:
        img[y0:y1, x0:x1] = cv2.addWeighted(panel, 0.4, panel, 0.0, 0)

    text_y = y0 + round(20 * k)
    for row in rows:
        for text, color, x in row:
            cv2.putText(img, text, (x0 + pad + x, text_y), FONT, font_scale, color, thick)
        text_y += row_h
