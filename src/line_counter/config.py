"""Processing configuration with the prototype defaults (standard library only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

#: Default COCO class IDs tracked and their display names.
DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

#: COCO class ID of "person" (face blurring only runs inside these boxes).
PERSON_CLASS_ID = 0


@dataclass
class ProcessingConfig:
    """Settings for detection, tracking, privacy and display.

    Attributes:
        model: Detector weights (Ultralytics official names auto-download).
        conf: Detection confidence threshold passed to ``track()``.
        imgsz: Inference image size passed to ``track()``.
        classes: COCO class IDs to track.
        class_names: Display names per class ID (falls back to model names).
        tracker: Ultralytics tracker config file.
        blur_faces: Whether to blur faces inside person boxes.
        face_model: Face detector weights used when ``blur_faces`` is True.
        face_conf: Confidence threshold of the face detector.
        blur_kernel: Gaussian blur kernel size (odd, odd).
        blur_sigma: Gaussian blur sigma.
        display: Show the live preview window (press Q to stop).
    """

    model: str = "weights/yolov8m.pt"
    conf: float = 0.01
    imgsz: int = 2048
    classes: Tuple[int, ...] = tuple(DEFAULT_CLASS_NAMES)
    class_names: Dict[int, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_NAMES))
    tracker: str = "botsort.yaml"
    blur_faces: bool = False
    face_model: str = "weights/face_yolov8m.pt"
    face_conf: float = 0.3
    blur_kernel: Tuple[int, int] = (51, 51)
    blur_sigma: float = 30.0
    display: bool = True

    def class_name(self, cls_id: int, model_names: Dict[int, str] | None = None) -> str:
        """Return the display name of a class ID.

        Args:
            cls_id: COCO class ID.
            model_names: Optional ``model.names`` mapping used as fallback.

        Returns:
            The configured name, the model name, or ``"Obj"``.
        """
        if cls_id in self.class_names:
            return self.class_names[cls_id]
        if model_names and cls_id in model_names:
            return str(model_names[cls_id]).capitalize()
        return "Obj"
