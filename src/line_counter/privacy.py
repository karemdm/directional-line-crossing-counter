"""Optional, best-effort face blurring inside person bounding boxes."""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0
        crop = frame[y1:y2, x1:x2]  # a view: edits go straight to the frame
        results = self.model(crop, verbose=False, conf=self.conf)
        blurred = 0
        for f_box in results[0].boxes:
            fx1, fy1, fx2, fy2 = map(int, f_box.xyxy[0].cpu().numpy())
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(crop.shape[1], fx2), min(crop.shape[0], fy2)
            if fx2 > fx1 and fy2 > fy1:
                roi = crop[fy1:fy2, fx1:fx2]
                crop[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(roi, self.kernel, self.sigma)
                blurred += 1
        return blurred
