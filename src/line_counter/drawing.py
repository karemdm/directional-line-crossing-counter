"""Frame annotations: corner-style boxes, counting lines, badges and the counts panel."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    k = max(0.5, img.shape[0] / 720)
    t = max(1, int(0.7 * k))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, t)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, t)
    cv2.putText(img, label, (x1, max(y1 - round(4 * k), round(10 * k))), FONT, max(0.3, 0.25 * k), color, 1)


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


#: Distinct BGR colors cycled over the counting lines.
LINE_PALETTE: List[Color] = [
    (0, 0, 255), (0, 165, 255), (0, 220, 255), (80, 200, 80), (255, 200, 0), (255, 100, 40),
    (200, 60, 200), (150, 150, 255), (60, 255, 180), (255, 150, 200), (0, 140, 90), (180, 120, 60),
]


def line_color(index: int) -> Color:
    """Return the palette color of the line at ``index`` (cycled)."""
    return LINE_PALETTE[index % len(LINE_PALETTE)]


def _scale(img: np.ndarray) -> float:
    return max(0.5, img.shape[0] / 720)


def _totals(line: CountingLine) -> Tuple[int, int]:
    n_in = sum(d.get(IN, 0) for d in line.counts.values())
    n_out = sum(d.get(OUT, 0) for d in line.counts.values())
    return n_in, n_out


def draw_counting_lines(img: np.ndarray, lines: Iterable[CountingLine], badges: bool = True) -> None:
    """Draw every counting line in its own color, with an optional count badge.

    Args:
        img: Frame to draw on (modified in place).
        lines: Counting lines in frame coordinates.
        badges: Draw a small pill ``name  N in / M out`` near each line midpoint
            (otherwise only the name at the first endpoint, as in the old HUD).
    """
    k = _scale(img)
    th = max(2, round(2 * k))
    lines = list(lines)
    for i, line in enumerate(lines):
        cv2.line(img, line.p1, line.p2, line_color(i), th)
    placed: List[Tuple[int, int, int, int]] = []
    for i, line in enumerate(lines):
        if badges:
            _draw_badge(img, line, line_color(i), k, placed)
        else:  # legacy: plain name label at the first endpoint
            cv2.putText(img, line.name, (line.p1[0], line.p1[1] - round(8 * k)), FONT, 0.6 * k,
                        line_color(i), th)


def _draw_badge(img: np.ndarray, line: CountingLine, color: Color, k: float,
                placed: List[Tuple[int, int, int, int]]) -> None:
    frame_h, frame_w = img.shape[:2]
    n_in, n_out = _totals(line)
    scale, thick = 0.45 * k, 1
    tail = f"  {n_in} in / {n_out} out"
    name = _ellipsize(line.name, scale, thick, max(int(0.4 * frame_w) - _text_width(tail, scale, thick), 20))
    text = name + tail
    (tw, th_), base = cv2.getTextSize(text, FONT, scale, thick)
    pad = round(4 * k)
    bw, bh = tw + 2 * pad, th_ + base + 2 * pad
    mx, my = (line.p1[0] + line.p2[0]) / 2, (line.p1[1] + line.p2[1]) / 2
    dx, dy = line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    nx, ny = -dy / length, dx / length
    off = abs(nx) * bw / 2 + abs(ny) * bh / 2 + round(6 * k)
    ux, uy = dx / length, dy / length
    step = bh + round(2 * k)
    best = None
    # Candidates: both sides of the line, then shifted along it; the first one
    # that does not overlap an earlier badge wins (else the least overlapping).
    for shift in (0, 1, -1, 2, -2, 3, -3):
        for sign in (1, -1):
            cx = mx + sign * nx * off + shift * step * ux
            cy = my + sign * ny * off + shift * step * uy
            x0 = int(min(max(cx - bw / 2, 0), max(frame_w - bw, 0)))
            y0 = int(min(max(cy - bh / 2, 0), max(frame_h - bh, 0)))
            miss = abs(x0 - (cx - bw / 2)) + abs(y0 - (cy - bh / 2))
            overlap = sum(max(0, min(x0 + bw, px1) - max(x0, px0)) * max(0, min(y0 + bh, py1) - max(y0, py0))
                          for px0, py0, px1, py1 in placed)
            score = (overlap > 0, overlap, miss > bw * 0.5, abs(shift))
            if best is None or score < best[0]:
                best = (score, x0, y0)
    _, x0, y0 = best
    placed.append((x0, y0, x0 + bw, y0 + bh))
    x1, y1 = min(x0 + bw, frame_w), min(y0 + bh, frame_h)
    roi = img[y0:y1, x0:x1]
    if roi.size:
        img[y0:y1, x0:x1] = (roi * 0.35).astype(np.uint8)  # darken: translucent dark pill
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), color, 1)
    cv2.putText(img, text, (x0 + pad, y0 + pad + th_), FONT, scale, TEXT_COLOR, thick)


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


PANEL_BG: Color = (28, 28, 28)
DIM_COLOR: Color = (130, 130, 130)
LAYOUT_SIDE = "side"
LAYOUT_OVERLAY = "overlay"
MIN_PANEL_W = 200
MIN_FONT = 0.3  # absolute smallest panel font scale (still readable)


def panel_width(frame_h: int, frame_w: Optional[int] = None) -> int:
    """Width of the side panel (400 px at 1080p), derived from the frame size.

    Scales with the height, at least ``MIN_PANEL_W`` px, and at most 40% of the
    frame width when it is given (so portrait frames do not get a huge panel).
    """
    w = max(MIN_PANEL_W, round(267 * max(frame_h / 720.0, 0.5)))
    if frame_w:
        w = min(w, max(MIN_PANEL_W, round(0.4 * frame_w)))
    return w


def output_size(width: int, height: int, layout: str) -> Tuple[int, int]:
    """Size of the written frames: widened by the side panel in ``side`` mode."""
    return (width + panel_width(height, width), height) if layout == LAYOUT_SIDE else (width, height)


def _ellipsize(text: str, scale: float, thickness: int, max_w: int) -> str:
    """Shorten ``text`` with ``...`` so that it fits ``max_w`` pixels."""
    if _text_width(text, scale, thickness) <= max_w:
        return text
    while len(text) > 1 and _text_width(text + "...", scale, thickness) > max_w:
        text = text[:-1]
    return text + "..."


def _class_entries(line: CountingLine, order: Sequence[str]) -> List[str]:
    names = list(order) + [c for c in line.counts if c not in order]  # classes outside the given order
    out = []
    for cls_name in names:
        data = line.counts.get(cls_name)
        if data and (data[IN] or data[OUT]):
            out.append(f"{cls_name} {data[IN]} in / {data[OUT]} out")
    return out


def _layout_blocks(lines: Sequence[CountingLine], order: Sequence[str], inner_w: int,
                   scale: float, thick: int):
    """Measure every block; returns ``(blocks, heights, row_h)``.

    A block is ``(line_index, [row texts])``; class entries share a row and wrap
    onto extra rows only when they exceed ``inner_w`` (an entry wider than a
    whole row is ellipsized). ``heights[i]`` is the pixel height of block ``i``.
    """
    sep = "   "
    row_h = round(cv2.getTextSize("Ag", FONT, scale, thick)[0][1] * 1.7)
    gap = round(row_h * 0.6)
    blocks, heights = [], []
    for i, line in enumerate(lines):
        entries = _class_entries(line, order)
        rows: List[str] = []
        if not entries:
            rows = ["no crossings yet"]
        for e in entries:
            e = _ellipsize(e, scale, thick, inner_w)
            if rows and _text_width(rows[-1] + sep + e, scale, thick) <= inner_w:
                rows[-1] += sep + e
            else:
                rows.append(e)
        blocks.append((i, rows))
        heights.append(row_h * (1 + len(rows)) + gap)
    return blocks, heights, row_h


def draw_side_panel(panel: np.ndarray, lines: Sequence[CountingLine],
                    class_order: Iterable[str]) -> None:
    """Fill ``panel`` (dark column) with one compact count block per line.

    Class names are drawn exactly as given. The font scale shrinks (down to
    ``MIN_FONT``) until all blocks fit; if they still do not, the blocks that fit
    are drawn followed by a dim ``+N more lines`` row. Nothing is drawn outside
    the panel.

    Args:
        panel: Panel image to draw on (modified in place).
        lines: Counting lines with their counts.
        class_order: Class names in display order.
    """
    h, w = panel.shape[:2]
    k = max(0.5, min(h / 720.0, w / 267.0))  # limited by both dimensions
    panel[:] = PANEL_BG
    pad = max(4, round(12 * k))
    inner_w = max(w - 2 * pad, 10)
    order = list(class_order)
    lines = list(lines)
    title_scale = 0.6 * k
    title_h = round(34 * k)
    footer_h = round(30 * k)
    avail = h - title_h - footer_h - pad
    scale = 0.5 * k
    min_scale = min(scale, MIN_FONT)
    thick = 1
    while True:
        blocks, heights, row_h = _layout_blocks(lines, order, inner_w, scale, thick)
        if sum(heights) <= avail or scale <= min_scale:
            break
        scale = max(min_scale, scale * 0.93)

    shown, used = 0, 0
    while shown < len(blocks) and used + heights[shown] <= avail:
        used += heights[shown]
        shown += 1
    hidden = len(blocks) - shown
    if hidden:  # keep one row for the "+N more lines" note
        while shown > 0 and used + row_h > avail:
            shown -= 1
            used -= heights[shown]
        hidden = len(blocks) - shown

    title = _ellipsize("Line counts", title_scale, max(1, round(k)), inner_w)
    cv2.putText(panel, title, (pad, pad + round(16 * k)), FONT, title_scale, TEXT_COLOR, max(1, round(k)))
    y = title_h + pad
    sw = max(5, round(20 * scale))
    tx = pad + sw + max(3, round(6 * k))
    text_w = max(w - tx - pad, 10)
    for i, rows in blocks[:shown]:
        cv2.rectangle(panel, (pad, y + 2), (pad + sw, y + 2 + sw), line_color(i), -1)
        cv2.putText(panel, _ellipsize(lines[i].name, scale, thick, text_w), (tx, y + sw + 2),
                    FONT, scale, TEXT_COLOR, thick)
        y += row_h
        for text in rows:
            dim = text == "no crossings yet"
            cv2.putText(panel, _ellipsize(text, scale * 0.95, thick, text_w), (tx, y + sw + 2), FONT,
                        scale * 0.95, DIM_COLOR if dim else MUTED_COLOR, thick)
            y += row_h
        y += round(row_h * 0.6)
    if hidden:
        cv2.putText(panel, _ellipsize(f"+{hidden} more lines", scale, thick, text_w), (tx, y + sw + 2),
                    FONT, scale, DIM_COLOR, thick)

    t_in = sum(d.get(IN, 0) for ln in lines for d in ln.counts.values())
    t_out = sum(d.get(OUT, 0) for ln in lines for d in ln.counts.values())
    foot = _ellipsize(f"Total: {t_in} in / {t_out} out", 0.45 * k, 1, inner_w)
    cv2.putText(panel, foot, (pad, h - max(3, round(10 * k))), FONT, 0.45 * k, TEXT_COLOR, 1)


def compose_frame(frame: np.ndarray, lines: Sequence[CountingLine], class_order: Iterable[str],
                  layout: str = LAYOUT_SIDE) -> np.ndarray:
    """Draw lines, badges and counts on a frame whose tracks are already drawn.

    Args:
        frame: Frame with tracked objects drawn (modified in place).
        lines: Counting lines with their counts.
        class_order: Class names in display order.
        layout: ``side`` (extra panel column, badges at the lines) or
            ``overlay`` (legacy HUD over the video, no badges).

    Returns:
        The frame to write: widened in ``side`` mode, the same image otherwise.
    """
    if layout == LAYOUT_SIDE:
        draw_counting_lines(frame, lines, badges=True)
        panel = np.empty((frame.shape[0], panel_width(frame.shape[0], frame.shape[1]), 3), dtype=np.uint8)
        draw_side_panel(panel, lines, class_order)
        return np.hstack([frame, panel])
    draw_counting_lines(frame, lines, badges=False)
    draw_hud(frame, lines, class_order)
    return frame
