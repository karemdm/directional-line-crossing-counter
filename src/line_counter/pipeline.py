"""Video processing loop: track, count, annotate and write frames."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2

from line_counter.config import PERSON_CLASS_ID, ProcessingConfig
from line_counter.counter import CountingLine, LineCounter
from line_counter.drawing import draw_counting_lines, draw_hud, draw_track
from line_counter.line_editor import window_size

logger = logging.getLogger(__name__)

PREVIEW_WINDOW = "Processing - Press 'Q' to Stop"
FALLBACK_FPS = 30.0


def default_output_path(input_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> Path:
    """Return ``<output_dir>/<input stem>_proc.mp4``.

    Args:
        input_path: Source video path.
        output_dir: Destination folder; defaults to the input folder.

    Returns:
        The output video path.
    """
    src = Path(input_path)
    folder = Path(output_dir) if output_dir else src.parent
    return folder / f"{src.stem}_proc.mp4"


def process_video(input_path: Union[str, Path], output_path: Union[str, Path],
                  lines: List[CountingLine], config: ProcessingConfig) -> LineCounter:
    """Run detection + tracking + counting over a video and save the result.

    Each annotated frame is written to disk as soon as it is processed, so a
    run stopped early (Q key or Ctrl+C) still leaves a playable partial video.

    Args:
        input_path: Source video path.
        output_path: Destination ``.mp4`` path.
        lines: Counting lines in original video pixel coordinates.
        config: Processing settings.

    Returns:
        The counter holding the final counts.

    Raises:
        RuntimeError: If the video cannot be opened or the writer fails.
    """
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    logger.info("Loading detector %s", config.model)
    model = YOLO(config.model)
    blurrer = None
    if config.blur_faces:
        from line_counter.privacy import FaceBlurrer

        blurrer = FaceBlurrer(config.face_model, config.face_conf, config.blur_kernel, config.blur_sigma)

    class_order = [config.class_name(c, model.names) for c in config.classes]
    fresh_lines = [CountingLine(ln.name, ln.p1, ln.p2) for ln in lines]  # start from zero
    counter = LineCounter(fresh_lines, class_order)

    if config.display:
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            results = model.track(
                frame,
                conf=config.conf,
                imgsz=config.imgsz,
                classes=list(config.classes),
                persist=True,
                verbose=False,
                tracker=config.tracker,
            )
            boxes = results[0].boxes if results else None
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int)
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                if blurrer is not None:  # blur before drawing so overlays stay sharp
                    for box, cls_id in zip(xyxy, cls_ids):
                        if cls_id == PERSON_CLASS_ID:
                            blurrer.blur_in_box(frame, box)

                for box, track_id, cls_id in zip(xyxy, ids, cls_ids):
                    x1, y1, x2, y2 = (int(v) for v in box)
                    cls_name = config.class_name(int(cls_id), model.names)
                    anchor = ((x1 + x2) // 2, y2)  # bottom-center ("feet" / wheels)
                    draw_track(frame, (x1, y1, x2, y2), int(track_id), cls_name, anchor)
                    for line_name, direction in counter.update(int(track_id), cls_name, anchor):
                        logger.info("Frame %d: #%d %s crossed %s (%s)",
                                    frame_idx, track_id, cls_name, line_name, direction)

            draw_counting_lines(frame, counter.lines)
            draw_hud(frame, counter.lines, class_order)
            out.write(frame)

            if total > 0 and frame_idx % 100 == 0:
                logger.info("Processed %d/%d frames", frame_idx, total)

            if config.display:
                cv2.imshow(PREVIEW_WINDOW, cv2.resize(frame, window_size(PREVIEW_WINDOW)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Stopped by user at frame %d", frame_idx)
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted at frame %d", frame_idx)
    finally:
        cap.release()
        out.release()
        if config.display:
            cv2.destroyAllWindows()

    logger.info("Saved %d frames to %s", frame_idx, output_path)
    for line_name, per_class in counter.summary().items():
        logger.info("%s: %s", line_name, per_class)
    return counter
