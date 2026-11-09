"""Frame annotations: corner-style boxes, counting lines and the counts HUD."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

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


def _put(img: np.ndarray, text: str, x: int, y: int, color: Color) -> None:
    cv2.putText(img, text, (x, y), FONT, 0.5, color, 1)


def draw_hud(img: np.ndarray, lines: Sequence[CountingLine], class_order: Iterable[str]) -> None:
    """Draw a translucent panel with per-line, per-class IN/OUT counts.

    Only classes with at least one crossing on a line are shown on that row,
    formatted as ``Line_1: Person[3 IN, 1 OUT]  Car[0 IN, 2 OUT]``.

    Args:
        img: Frame to draw on (modified in place).
        lines: Counting lines with their counts.
        class_order: Class names in display order.
    """
    order = list(class_order)
    panel_height = 30 + len(lines) * 25
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (550, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    text_y = 30
    for line in lines:
        x = 20
        title = f"{line.name}: "
        _put(img, title, x, text_y, TEXT_COLOR)
        x += len(title) * 9
        for cls_name in order:
            data = line.counts.get(cls_name)
            if not data or (data[IN] == 0 and data[OUT] == 0):
                continue
            label_cls = f"{cls_name}["
            _put(img, label_cls, x, text_y, MUTED_COLOR)
            x += len(label_cls) * 8
            label_in = f"{data[IN]} IN"
            _put(img, label_in, x, text_y, IN_COLOR)
            x += len(label_in) * 8 + 4
            _put(img, ",", x - 3, text_y, MUTED_COLOR)
            x += 8
            label_out = f"{data[OUT]} OUT"
            _put(img, label_out, x, text_y, OUT_COLOR)
            x += len(label_out) * 8 + 4
            _put(img, "]", x - 2, text_y, MUTED_COLOR)
            x += 15
        text_y += 22
