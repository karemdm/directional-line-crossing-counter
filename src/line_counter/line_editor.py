"""Interactive OpenCV editor for drawing counting lines on a frame.

Click two points to create a line; repeat for more lines; press ENTER or ESC
to finish. The window can be resized or maximised: clicks are mapped back to
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


class _LineEditor:
    """Mouse-callback state for :func:`draw_lines`."""

    def __init__(self, frame: np.ndarray, lines: List[CountingLine]) -> None:
        self.frame = frame
        self.lines = lines
        self.pending: List[Tuple[int, int]] = []

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        """Handle a mouse event and redraw the scaled canvas."""
        orig_h, orig_w = self.frame.shape[:2]
        win_w, win_h = window_size(WINDOW_NAME)
        scale_x, scale_y = orig_w / win_w, orig_h / win_h

        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending.append((int(x * scale_x), int(y * scale_y)))
            if len(self.pending) == 2:
                name = f"Line_{len(self.lines) + 1}"
                self.lines.append(CountingLine(name, self.pending[0], self.pending[1]))
                logger.info("Added %s: %s -> %s", name, self.pending[0], self.pending[1])
                self.pending = []

        canvas = cv2.resize(self.frame, (win_w, win_h))

        def to_win(pt: Tuple[int, int]) -> Tuple[int, int]:
            return int(pt[0] / scale_x), int(pt[1] / scale_y)

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
    editor.on_mouse(cv2.EVENT_MOUSEMOVE, 0, 0, 0, None)  # initial render
    try:
        while (cv2.waitKey(20) & 0xFF) not in _FINISH_KEYS:
            pass
    finally:
        cv2.destroyWindow(WINDOW_NAME)
    return lines
