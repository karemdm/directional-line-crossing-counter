"""Tests for the counts HUD."""

import numpy as np

from line_counter.counter import IN, OUT, CountingLine
from line_counter.drawing import draw_hud

CLASSES = ["Person", "Bicycle", "Motorcycle", "Car", "Bus", "Truck"]


def _lines(n: int) -> list:
    lines = []
    for i in range(n):
        line = CountingLine(name=f"Line_{i + 1}", p1=(0, 0), p2=(10, 10))
        for c in CLASSES:
            line.counts[c] = {IN: 12 + i, OUT: 3}
        lines.append(line)
    return lines


def test_hud_12_lines_fits_1080p_frame():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    draw_hud(img, _lines(12), CLASSES)
    ys, xs = np.nonzero(img.any(axis=2))
    assert xs.size > 0
    assert xs.max() < 1920 and xs.min() >= 10
    assert ys.max() < 1080 // 2  # does not cover most of the frame


def test_hud_wraps_on_narrow_frame_without_overflow():
    img = np.zeros((720, 400, 3), dtype=np.uint8)
    draw_hud(img, _lines(2), CLASSES)
    assert img.any()
    assert not img[:, 390:].any()  # right margin untouched


def test_hud_skips_empty_lines():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_hud(img, [CountingLine(name="L", p1=(0, 0), p2=(1, 1))], CLASSES)
    assert img.any()  # only the title


# --- side panel layout -------------------------------------------------------

import pytest

from line_counter.drawing import (LINE_PALETTE, PANEL_BG, compose_frame, draw_counting_lines,
                                  draw_corner_box, line_color, output_size, panel_width)


def _spread_lines(n: int) -> list:
    lines = _lines(n)
    for i, ln in enumerate(lines):
        ln.p1, ln.p2 = (100 + 120 * i, 200 + 30 * i), (160 + 120 * i, 700)
    return lines


def test_side_layout_widens_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = compose_frame(frame, _spread_lines(12), CLASSES, "side")
    assert panel_width(1080) == 400
    assert out.shape == (1080, 1920 + 400, 3)
    assert output_size(1920, 1080, "side") == (2320, 1080)
    assert output_size(1920, 1080, "overlay") == (1920, 1080)


def test_overlay_layout_keeps_size():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert compose_frame(frame, _spread_lines(2), CLASSES, "overlay").shape == (720, 1280, 3)


def test_side_panel_fits_12_lines_at_1080p():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = compose_frame(frame, _spread_lines(12), CLASSES, "side")
    panel = out[:, 1920:]
    content = (panel != np.array(PANEL_BG, dtype=np.uint8)).any(axis=2)
    assert content.any()
    assert not content[:, -4:].any()  # nothing clipped at the right edge
    rows = np.nonzero(content.any(axis=1))[0]
    assert rows.max() < 1080 - 1
    # the last line's swatch (its palette color) is drawn inside the panel
    assert (panel == np.array(line_color(11), dtype=np.uint8)).all(axis=2).any()


def test_video_area_has_no_counts_text_in_side_mode():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    lines = [CountingLine(name="L", p1=(600, 100), p2=(600, 600))]
    lines[0].counts["Person"] = {IN: 5, OUT: 2}
    out = compose_frame(frame, lines, CLASSES, "side")
    assert not out[:, :300, :1280].any()  # far from the line/badge: untouched


@pytest.mark.parametrize("p1,p2", [((5, 5), (60, 5)), ((1900, 1070), (1915, 700)),
                                   ((0, 540), (0, 900)), ((960, 0), (1000, 1079))])
def test_badges_stay_inside_frame(p1, p2):
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    line = CountingLine(name="Line_4", p1=p1, p2=p2)
    line.counts["Person"] = {IN: 12, OUT: 8}
    draw_counting_lines(img, [line], badges=True)  # would raise on out-of-range writes
    assert img.any()
    assert img.shape == (1080, 1920, 3)


def test_line_colors_distinct_and_cycled():
    assert len(set(LINE_PALETTE)) == len(LINE_PALETTE)
    assert line_color(len(LINE_PALETTE)) == line_color(0)


def test_track_labels_are_thin():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    draw_corner_box(img, (100, 100), (200, 300), (0, 255, 0), "#1 Person")
    assert (img[:, :, 1] > 0).sum() < 1500


# --- generic layout: any classes, line counts and frame sizes -------------------

LONG_CLASSES = [f"a_very_long_custom_class_name_number_{i}" for i in range(15)]


def _generic_lines(n, classes):
    lines = []
    for i in range(n):
        ln = CountingLine(name=f"Entrance_with_a_really_long_name_{i + 1}", p1=(20 + i, 20), p2=(60 + i, 200))
        for c in classes:
            ln.counts[c] = {IN: 100 + i, OUT: 7}
        lines.append(ln)
    return lines


def _check_bounds(h, w, lines, classes):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    out = compose_frame(frame, lines, classes, "side")
    pw = panel_width(h, w)
    assert out.shape == (h, w + pw, 3)
    panel = out[:, w:]
    content = (panel != np.array(PANEL_BG, dtype=np.uint8)).any(axis=2)
    assert content.any()
    assert not content[:, -2:].any()  # nothing touching the right edge
    return out


def test_many_long_class_names_stay_in_panel():
    _check_bounds(1080, 1920, _generic_lines(4, LONG_CLASSES), LONG_CLASSES)


def test_30_lines_show_more_lines_note():
    out = _check_bounds(1080, 1920, _generic_lines(30, CLASSES), CLASSES)
    assert out.any()


@pytest.mark.parametrize("h,w", [(360, 640), (1920, 1080), (2160, 3840), (240, 320)])
def test_sizes_and_aspect_ratios(h, w):
    out = _check_bounds(h, w, _generic_lines(12, CLASSES), CLASSES)
    assert panel_width(h, w) <= max(200, round(0.4 * w))
    assert out.shape[0] == h


def test_overflow_note_when_blocks_do_not_fit(monkeypatch):
    import line_counter.drawing as d
    seen = []
    orig = d.cv2.putText

    def spy(img, text, *a, **k):
        seen.append(text)
        return orig(img, text, *a, **k)

    monkeypatch.setattr(d.cv2, "putText", spy)
    compose_frame(np.zeros((360, 640, 3), dtype=np.uint8), _generic_lines(30, CLASSES), CLASSES, "side")
    assert any(t.startswith("+") and t.endswith("more lines") for t in seen)


def test_class_names_are_drawn_as_given_and_extra_classes_shown(monkeypatch):
    import line_counter.drawing as d
    seen = []
    orig = d.cv2.putText
    monkeypatch.setattr(d.cv2, "putText", lambda img, text, *a, **k: (seen.append(text), orig(img, text, *a, **k))[1])
    ln = CountingLine(name="L", p1=(5, 5), p2=(50, 50))
    ln.counts["weird-Class_9"] = {IN: 1, OUT: 2}  # not in the class order
    compose_frame(np.zeros((720, 1280, 3), dtype=np.uint8), [ln], [], "side")
    assert any("weird-Class_9 1 in / 2 out" in t for t in seen)
    assert any(t.startswith("Total: 1 in / 2 out") for t in seen)
