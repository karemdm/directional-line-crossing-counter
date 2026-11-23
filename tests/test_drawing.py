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
