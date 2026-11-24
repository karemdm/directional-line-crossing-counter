"""Tests for the line editor's coordinate mapping and event handling (no real window)."""

import cv2
import numpy as np
import pytest

from line_counter import line_editor
from line_counter.counter import CountingLine
from line_counter.line_editor import _LineEditor, frame_to_window, window_to_frame

FRAME = (1920, 1080)  # (width, height)


def test_window_to_frame_scales_each_axis_independently():
    assert window_to_frame(480, 135, (960, 540), FRAME) == (960, 270)
    assert window_to_frame(0, 0, (960, 540), FRAME) == (0, 0)
    assert window_to_frame(100, 50, (3840, 2160), FRAME) == (50, 25)  # window larger than the video
    assert window_to_frame(100, 50, (400, 100), FRAME) == (480, 540)  # stretched (non-uniform) window


def test_window_to_frame_clamps_to_the_frame():
    assert window_to_frame(-20, -5, (960, 540), FRAME) == (0, 0)
    assert window_to_frame(5000, 5000, (960, 540), FRAME) == (1919, 1079)
    assert window_to_frame(960, 540, (960, 540), FRAME) == (1919, 1079)  # one past the last pixel


def test_frame_to_window_inverts_the_mapping_within_a_pixel():
    for win in ((960, 540), (1280, 720), (700, 900)):
        for pt in ((0, 0), (1919, 1079), (1000, 500), (3, 617)):
            x, y = frame_to_window(pt, win, FRAME)
            bx, by = window_to_frame(x, y, win, FRAME)
            assert abs(bx - pt[0]) <= FRAME[0] / win[0] and abs(by - pt[1]) <= FRAME[1] / win[1]


@pytest.fixture
def editor(monkeypatch):
    shown = []
    monkeypatch.setattr(line_editor, "window_size", lambda name: (960, 540))
    monkeypatch.setattr(cv2, "imshow", lambda name, img: shown.append(img.shape))
    frame = np.zeros((1080, 1920, 3), np.uint8)
    ed = _LineEditor(frame, [])
    ed.shown = shown
    return ed


def click(ed, x, y):
    ed.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)


def test_two_clicks_make_a_line_in_original_coordinates(editor):
    click(editor, 100, 100)
    assert editor.pending == [(200, 200)] and editor.lines == []
    click(editor, 400, 300)
    assert [(ln.name, ln.p1, ln.p2) for ln in editor.lines] == [("Line_1", (200, 200), (800, 600))]
    assert editor.pending == []
    assert editor.shown[-1] == (540, 960, 3)  # canvas has the window size


def test_clicking_the_same_point_twice_does_not_create_a_zero_length_line(editor):
    click(editor, 100, 100)
    click(editor, 100, 100)
    assert editor.lines == [] and editor.pending == [(200, 200)]


def test_clicks_outside_the_window_are_clamped_into_the_frame(editor):
    click(editor, -10, 10)
    click(editor, 2000, 2000)
    (line,) = editor.lines
    assert line.p1 == (0, 20) and line.p2 == (1919, 1079)


def test_mouse_move_only_redraws(editor):
    editor.on_mouse(cv2.EVENT_MOUSEMOVE, 10, 10, 0, None)
    assert editor.pending == [] and editor.lines == [] and editor.shown


def test_new_lines_continue_the_numbering_of_existing_ones(monkeypatch):
    monkeypatch.setattr(line_editor, "window_size", lambda name: (1920, 1080))
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    ed = _LineEditor(np.zeros((1080, 1920, 3), np.uint8), [CountingLine("Line_1", (0, 0), (5, 5))])
    click(ed, 1, 1)
    click(ed, 9, 9)
    assert [ln.name for ln in ed.lines] == ["Line_1", "Line_2"]


def test_draw_lines_returns_when_the_window_is_closed_with_the_x_button(monkeypatch):
    calls = {"waitKey": 0}

    def wait_key(ms):
        calls["waitKey"] += 1
        return -1  # no key is ever pressed

    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(line_editor, "window_size", lambda name: (100, 100))
    # visible for the first two polls, then the user closes the window
    states = iter([1.0, 1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(cv2, "getWindowProperty", lambda name, prop: next(states))
    lines = line_editor.draw_lines(np.zeros((50, 50, 3), np.uint8))
    assert lines == [] and calls["waitKey"] < 10


def test_window_closed_helper(monkeypatch):
    monkeypatch.setattr(cv2, "getWindowProperty", lambda n, p: 1.0)
    assert line_editor.window_closed("w") is False
    monkeypatch.setattr(cv2, "getWindowProperty", lambda n, p: 0.0)
    assert line_editor.window_closed("w") is True

    def boom(n, p):
        raise cv2.error("no such window")

    monkeypatch.setattr(cv2, "getWindowProperty", boom)
    assert line_editor.window_closed("w") is True


def test_is_quit_key_accepts_both_cases():
    from line_counter.pipeline import is_quit_key

    assert is_quit_key(ord("q")) and is_quit_key(ord("Q"))
    assert not is_quit_key(-1) and not is_quit_key(ord("a"))
    assert is_quit_key(ord("q") | 0x100000)  # high bits set by some backends are ignored


def test_undo_removes_the_pending_point_then_the_last_line(editor):
    click(editor, 100, 100)
    click(editor, 400, 300)
    click(editor, 50, 50)
    assert editor.undo() is True and editor.pending == [] and len(editor.lines) == 1
    assert editor.undo() is True and editor.lines == []
    assert editor.undo() is False


def test_undo_then_new_line_reuses_the_name(editor):
    click(editor, 100, 100)
    click(editor, 400, 300)
    editor.undo()
    click(editor, 10, 10)
    click(editor, 20, 20)
    assert [ln.name for ln in editor.lines] == ["Line_1"]


def test_draw_lines_handles_undo_keys_and_redraws_on_resize(monkeypatch):
    shown = []
    state = {"size": (100, 100)}
    keys = iter([ord("u"), -1, 8, 13])

    def wait_key(ms):
        k = next(keys)
        if k == -1:
            state["size"] = (200, 200)  # the user resizes the window
        return k

    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, img: shown.append(img.shape[:2]))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda n, p: 1.0)
    monkeypatch.setattr(line_editor, "window_size", lambda name: state["size"])
    existing = [CountingLine("Line_1", (0, 0), (5, 5)), CountingLine("Line_2", (1, 1), (6, 6))]
    line_editor.draw_lines(np.zeros((50, 50, 3), np.uint8), existing)
    assert existing == []  # two undo keys removed both lines
    assert (200, 200) in shown  # redrawn at the new window size
