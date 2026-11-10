"""Optional, best-effort face blurring inside person bounding boxes."""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Maximum number of person crops sent to the face model in one call.
FACE_BATCH_SIZE = 16


class FaceBlurrer:
    """Detect faces inside person crops with a YOLO face model and blur them.

    Faces that the detector misses (small, occluded, profile, blurred by
    motion) are NOT blurred. Treat the output as best-effort anonymisation.

    Args:
        model_path: Path to the face detector weights.
        conf: Face detection confidence threshold.
        kernel: Gaussian blur kernel size (odd, odd).
        sigma: Gaussian blur sigma.
    """

    def __init__(self, model_path: str, conf: float = 0.3,
                 kernel: Tuple[int, int] = (51, 51), sigma: float = 30.0) -> None:
        from ultralytics import YOLO

        logger.info("Loading face model %s", model_path)
        self.model = YOLO(model_path)
        self.conf = conf
        self.kernel = kernel
        self.sigma = sigma

    def blur_in_box(self, frame: np.ndarray, box: Sequence[int]) -> int:
        """Blur faces found inside one person box, in place.

        Args:
            frame: Full BGR frame (modified in place).
            box: Person box ``(x1, y1, x2, y2)`` in frame pixels.

        Returns:
            Number of face regions blurred.
        """
        return self.blur_in_boxes(frame, [box])

    def blur_in_boxes(self, frame: np.ndarray, boxes: Iterable[Sequence[int]]) -> int:
        """Blur faces found inside several person boxes, in place.

        The crops are sent to the face model in batches (one call per
        ``FACE_BATCH_SIZE`` crops) instead of one call per person.

        Args:
            frame: Full BGR frame (modified in place).
            boxes: Person boxes ``(x1, y1, x2, y2)`` in frame pixels.

        Returns:
            Number of face regions blurred.
        """
        h, w = frame.shape[:2]
        crops: List[np.ndarray] = []
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crops.append(frame[y1:y2, x1:x2])  # views: edits go straight to the frame
        blurred = 0
        for start in range(0, len(crops), FACE_BATCH_SIZE):
            batch = crops[start:start + FACE_BATCH_SIZE]
            results = self.model(batch, verbose=False, conf=self.conf)
            for crop, result in zip(batch, results):
                blurred += self._blur_faces(crop, result.boxes.xyxy.cpu().numpy())
        return blurred

    def _blur_faces(self, crop: np.ndarray, face_boxes: np.ndarray) -> int:
        """Blur the given face boxes (crop coordinates) inside ``crop``."""
        ch, cw = crop.shape[:2]
        blurred = 0
        for fx1, fy1, fx2, fy2 in face_boxes.astype(int):
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(cw, fx2), min(ch, fy2)
            if fx2 > fx1 and fy2 > fy1:
                roi = crop[fy1:fy2, fx1:fx2]
                crop[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(roi, self.kernel, self.sigma)
                blurred += 1
        return blurred
