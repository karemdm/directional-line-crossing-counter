"""Interactive OpenCV editor for drawing counting lines on a frame.

Click two points to create a line; repeat for more lines; press ENTER or ESC
to finish; press U or BACKSPACE to undo the last point or line. The window can be resized or maximised: clicks are mapped back to
the original video resolution.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from line_counter.counter import CountingLine

logger = logging.getLogger(__name__)

WINDOW_NAME = "Draw Lines - Press [ENTER] to Finish"
LINE_COLOR = (0, 0, 255)
PENDING_COLOR = (0, 255, 0)
_FINISH_KEYS = (13, 27)  # ENTER, ESC
_UNDO_KEYS = (ord("u"), ord("U"), 8)  # U, BACKSPACE


def read_first_frame(video_path: str) -> Optional[np.ndarray]:
    """Read the first frame of a video.

    Args:
        video_path: Path to the video file.

    Returns:
        The BGR frame, or None if it could not be read.
    """
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def window_size(window_name: str) -> Tuple[int, int]:
    """Return the current image area size of an OpenCV window (at least 1x1).

    Args:
        window_name: Name of the OpenCV window.

    Returns:
        ``(width, height)`` in screen pixels.
    """
    rect = cv2.getWindowImageRect(window_name)
    return max(rect[2], 1), max(rect[3], 1)


def window_closed(window_name: str) -> bool:
    """Tell whether the user closed an OpenCV window (X button) or it does not exist."""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def window_to_frame(x: int, y: int, win: Tuple[int, int], frame: Tuple[int, int]) -> Tuple[int, int]:
    """Map a window pixel to frame pixel coordinates, clamped to the frame.

    Args:
        x: Window x.
        y: Window y.
        win: ``(width, height)`` of the window image area.
        frame: ``(width, height)`` of the original frame.

    Returns:
        ``(x, y)`` in original pixels, inside ``[0, w-1] x [0, h-1]`` (a click
        can be reported outside the window while the button is held).
    """
    px = int(x * frame[0] / win[0])
    py = int(y * frame[1] / win[1])
    return min(max(px, 0), frame[0] - 1), min(max(py, 0), frame[1] - 1)


def frame_to_window(pt: Tuple[int, int], win: Tuple[int, int], frame: Tuple[int, int]) -> Tuple[int, int]:
    """Map frame pixel coordinates to window pixels (inverse of :func:`window_to_frame`)."""
    return int(pt[0] * win[0] / frame[0]), int(pt[1] * win[1] / frame[1])


class _LineEditor:
    """Mouse-callback state for :func:`draw_lines`."""

    def __init__(self, frame: np.ndarray, lines: List[CountingLine]) -> None:
        self.frame = frame
        self.lines = lines
        self.pending: List[Tuple[int, int]] = []
        self.last_size: Tuple[int, int] = (0, 0)

    def undo(self) -> bool:
        """Remove the pending first point, or else the last line.

        Returns:
            True if something was removed.
        """
        if self.pending:
            self.pending.pop()
            return True
        if self.lines:
            removed = self.lines.pop()
            logger.info("Removed %s", removed.name)
            return True
        return False

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        """Handle a mouse event and redraw the scaled canvas."""
        orig_h, orig_w = self.frame.shape[:2]
        win_w, win_h = window_size(WINDOW_NAME)
        frame_size = (orig_w, orig_h)

        if event == cv2.EVENT_LBUTTONDOWN:
            point = window_to_frame(x, y, (win_w, win_h), frame_size)
            if self.pending and self.pending[0] == point:
                logger.info("Second point equals the first one; click a different point")
            else:
                self.pending.append(point)
            if len(self.pending) == 2:
                name = f"Line_{len(self.lines) + 1}"
                self.lines.append(CountingLine(name, self.pending[0], self.pending[1]))
                logger.info("Added %s: %s -> %s", name, self.pending[0], self.pending[1])
                self.pending = []
        self.render()

    def render(self) -> None:
        """Redraw the frame, lines and pending point at the current window size."""
        orig_h, orig_w = self.frame.shape[:2]
        win_w, win_h = window_size(WINDOW_NAME)
        frame_size = (orig_w, orig_h)
        self.last_size = (win_w, win_h)
        canvas = cv2.resize(self.frame, (win_w, win_h))

        def to_win(pt: Tuple[int, int]) -> Tuple[int, int]:
            return frame_to_window(pt, (win_w, win_h), frame_size)

        for line in self.lines:
            p1, p2 = to_win(line.p1), to_win(line.p2)
            cv2.line(canvas, p1, p2, LINE_COLOR, 2)
            cv2.putText(canvas, line.name, (p1[0], p1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_COLOR, 1)
        if len(self.pending) == 1:
            cv2.circle(canvas, to_win(self.pending[0]), 4, PENDING_COLOR, -1)
        cv2.imshow(WINDOW_NAME, canvas)


def draw_lines(frame: np.ndarray, lines: Optional[List[CountingLine]] = None) -> List[CountingLine]:
    """Open the editor window and let the user draw counting lines.

    Args:
        frame: Frame to draw on (original resolution, BGR).
        lines: Existing lines to show and append to (modified in place).

    Returns:
        The list of lines after editing.
    """
    lines = [] if lines is None else lines
    editor = _LineEditor(frame, lines)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, editor.on_mouse)
    editor.render()
    try:
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in _FINISH_KEYS:
                break
            if window_closed(WINDOW_NAME):  # closed with the X button: finish with what was drawn
                break
            if key in _UNDO_KEYS:
                editor.undo()
                editor.render()
            elif window_size(WINDOW_NAME) != editor.last_size:  # window resized: redraw
                editor.render()
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:  # already gone (closed by the user)
            pass
    return lines
