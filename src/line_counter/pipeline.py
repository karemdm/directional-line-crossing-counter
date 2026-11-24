"""Video processing loop: track, count, annotate and write frames."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2

from line_counter.config import PERSON_CLASS_ID, ProcessingConfig
from line_counter.counter import CountingLine, LineCounter
from line_counter.drawing import compose_frame, draw_track, output_size
from line_counter.geometry import segment_misses_frame
from line_counter.line_editor import window_closed, window_size
from line_counter.tracks_io import (infer_frame_step, load_tracks, track_record,
                                    write_frame_tracks)

logger = logging.getLogger(__name__)

PREVIEW_WINDOW = "Processing - Press 'Q' to Stop"
FALLBACK_FPS = 30.0


def is_quit_key(key: int) -> bool:
    """Tell whether a ``cv2.waitKey`` result is Q or q (high bits are ignored)."""
    return (key & 0xFF) in (ord("q"), ord("Q"))


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
        RuntimeError: If the video cannot be opened, no frame can be read or
            the writer fails.
        ValueError: If a ``--replay-tracks`` file is invalid.
        OSError: If the tracks file cannot be created.
    """
    replay = None
    model = None
    if config.replay_tracks:
        replay = load_tracks(config.replay_tracks)  # no detector, tracker or face blur needed
    else:
        from ultralytics import YOLO

        # Load the models first, so a bad weights path does not leave an open
        # capture or an empty output file behind.
        logger.info("Loading detector %s", config.model)
        model = YOLO(config.model)
    blurrer = None
    if config.blur_faces and replay is None:
        from line_counter.privacy import FaceBlurrer

        blurrer = FaceBlurrer(config.face_model, config.face_conf, config.blur_kernel, config.blur_sigma)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (fps and fps > 0):  # 0, negative or NaN from some containers
        fps = FALLBACK_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read the frame size of: {input_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    step = max(1, int(config.frame_step))
    if replay is not None:
        step = infer_frame_step(list(replay), step)  # same step as the saved run
    out_w, out_h = output_size(w, h, config.layout)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    if replay is not None:  # class names come from the saved tracks, ordered by class id
        saved = {t["cls"]: t["name"] for ts in replay.values() for t in ts}
        class_order = [saved[c] for c in sorted(saved)]
    else:
        class_order = [config.class_name(c, model.names) for c in config.classes]
    class_order = list(dict.fromkeys(class_order))  # two class IDs can share a display name
    fresh_lines = [CountingLine(ln.name, ln.p1, ln.p2,
                                cooldown_frames=config.cooldown_frames,
                                cooldown_px=config.cooldown_px) for ln in lines]  # start from zero
    counter = LineCounter(fresh_lines, class_order)
    for ln in fresh_lines:
        if segment_misses_frame(ln.p1, ln.p2, w, h):
            logger.warning("%s lies completely outside the %dx%d frame and can never count; "
                           "were the lines drawn on a video of another resolution?", ln.name, w, h)

    if config.display:
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

    tracks_fh = None
    frame_idx = 0
    processed = 0
    last_replay_frame = max(replay) if replay else 0
    try:
        if config.save_tracks and replay is None:
            Path(config.save_tracks).parent.mkdir(parents=True, exist_ok=True)
            tracks_fh = open(config.save_tracks, "w", encoding="utf-8")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if replay is not None:
                if frame_idx > last_replay_frame:
                    break  # nothing left to render; do not decode the rest of the video
                if frame_idx not in replay:  # replay uses the frame indices of the file
                    continue
                tracks = [(t["id"], t["cls"], t["name"], t.get("conf", 0.0), t["box"])
                          for t in replay[frame_idx]]
            else:
                if (frame_idx - 1) % step:  # --frame-step: skip frames (decoded but not processed)
                    continue
                tracks = _detect(model, frame, config, blurrer)
                if tracks_fh is not None:
                    write_frame_tracks(tracks_fh, frame_idx, [track_record(*t) for t in tracks])
            for track_id, _cls_id, cls_name, _conf, box in tracks:
                x1, y1, x2, y2 = (int(v) for v in box)
                anchor = ((x1 + x2) // 2, y2)  # bottom-center ("feet" / wheels)
                draw_track(frame, (x1, y1, x2, y2), int(track_id), cls_name, anchor)
                for line_name, direction in counter.update(int(track_id), cls_name, anchor, frame_idx):
                    logger.info("Frame %d: #%d %s crossed %s (%s)",
                                frame_idx, track_id, cls_name, line_name, direction)

            processed += 1
            frame = compose_frame(frame, counter.lines, class_order, config.layout)
            if frame.shape[1] != out_w or frame.shape[0] != out_h:  # writer silently drops mismatched frames
                frame = cv2.resize(frame, (out_w, out_h))
            out.write(frame)

            if total > 0 and frame_idx % 100 == 0:
                logger.info("Processed %d/%d frames", frame_idx, total)

            if config.display:
                cv2.imshow(PREVIEW_WINDOW, cv2.resize(frame, window_size(PREVIEW_WINDOW)))
                if is_quit_key(cv2.waitKey(1)) or window_closed(PREVIEW_WINDOW):
                    logger.info("Stopped by user at frame %d", frame_idx)
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted at frame %d", frame_idx)
    finally:
        cap.release()
        out.release()
        if tracks_fh is not None:
            tracks_fh.close()
        if config.display:
            cv2.destroyAllWindows()

    if frame_idx == 0:
        raise RuntimeError(f"No frames could be read from: {input_path}")
    if processed == 0:
        logger.warning("No frame was processed; do the tracks file and the video belong together?")
    logger.info("Saved %d frames to %s", processed, output_path)
    for line_name, per_class in counter.summary().items():
        logger.info("%s: %s", line_name, per_class)
    return counter


def _detect(model, frame, config: ProcessingConfig, blurrer) -> list:
    """Run detection + tracking on one frame (blurring person boxes if asked).

    Returns:
        ``(id, cls_id, cls_name, conf, box)`` for detections that have a tracker ID.
    """
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
    if boxes is None or not len(boxes):
        return []
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    # Blur before drawing so overlays stay sharp. This also covers
    # frames where the tracker returns detections without IDs.
    if blurrer is not None:
        person_boxes = xyxy[cls_ids == PERSON_CLASS_ID]
        if len(person_boxes):
            blurrer.blur_in_boxes(frame, person_boxes)
    if boxes.id is None:
        return []
    ids = boxes.id.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    return [(int(i), int(c), config.class_name(int(c), model.names), float(cf), b)
            for b, i, c, cf in zip(xyxy, ids, cls_ids, confs)]
