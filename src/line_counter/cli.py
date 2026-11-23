"""Command-line interface (``line-counter``)."""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Sequence

from line_counter import __version__
from line_counter.config import DEFAULT_CLASS_NAMES, ProcessingConfig

logger = logging.getLogger("line_counter")

_DEFAULTS = ProcessingConfig()


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        The configured parser.
    """
    p = argparse.ArgumentParser(
        prog="line-counter",
        description="Count people and vehicles crossing user-drawn lines in a video, "
                    "by direction (IN/OUT). Without --input the Tkinter launcher opens.",
    )
    p.add_argument("--input", "-i", help="Input video file. If omitted, the GUI launcher opens.")
    p.add_argument("--output", "-o",
                   help="Output video path (default: <input folder>/<input name>_proc.mp4).")
    p.add_argument("--lines", help="JSON file with counting lines; skips interactive drawing.")
    p.add_argument("--save-lines", help="Write the interactively drawn lines to this JSON file.")
    p.add_argument("--model", default=_DEFAULTS.model,
                   help="Detector weights; Ultralytics official names auto-download "
                        "(default: %(default)s).")
    p.add_argument("--conf", type=float, default=_DEFAULTS.conf,
                   help="Detection confidence threshold (default: %(default)s; very permissive, "
                        "0.1-0.25 is a common range).")
    p.add_argument("--imgsz", type=int, default=_DEFAULTS.imgsz,
                   help="Inference image size in pixels (default: %(default)s).")
    p.add_argument("--classes", type=int, nargs="+", default=list(_DEFAULTS.classes),
                   help="COCO class IDs to track (default: %(default)s = "
                        + ", ".join(DEFAULT_CLASS_NAMES.values()) + ").")
    p.add_argument("--tracker", default=_DEFAULTS.tracker,
                   help="Ultralytics tracker config, e.g. botsort.yaml or bytetrack.yaml "
                        "(default: %(default)s).")
    p.add_argument("--frame-step", type=_positive_int, default=_DEFAULTS.frame_step,
                   help="Process every Nth frame (default: %(default)s = all). The output is "
                        "written at fps/N, so it plays at the original speed.")
    p.add_argument("--cooldown-frames", type=_non_negative_int, default=_DEFAULTS.cooldown_frames,
                   help="Do not count a crossing in the same direction within this many source "
                        "frames and --cooldown-px pixels of a previous one on the same line, "
                        "to absorb ID switches (default: %(default)s = off).")
    p.add_argument("--cooldown-px", type=float, default=_DEFAULTS.cooldown_px,
                   help="Radius in pixels for --cooldown-frames (default: %(default)s).")
    p.add_argument("--layout", choices=("side", "overlay"), default=_DEFAULTS.layout,
                   help="side: counts in a panel next to the video (output is wider); overlay: "
                        "counts drawn over the video (default: %(default)s).")
    p.add_argument("--save-tracks", metavar="PATH.jsonl",
                   help="Write the tracks of every processed frame to this JSON Lines file.")
    p.add_argument("--replay-tracks", metavar="PATH.jsonl",
                   help="Skip detection/tracking and re-render from a file written by "
                        "--save-tracks (face blur is not applied in replay).")
    p.add_argument("--save-counts", metavar="PATH.json",
                   help="Write the final per-line, per-class IN/OUT counts (plus input name, "
                        "lines and main settings) to this JSON file when processing ends.")
    p.add_argument("--blur-faces", action="store_true",
                   help="Blur faces detected inside person boxes (best-effort).")
    p.add_argument("--face-model", default=_DEFAULTS.face_model,
                   help="Face detector weights used with --blur-faces (default: %(default)s).")
    p.add_argument("--no-display", action="store_true",
                   help="Do not show the live preview window (stop with Ctrl+C).")
    p.add_argument("--gui", action="store_true", help="Open the Tkinter launcher.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    """Build a :class:`ProcessingConfig` from parsed arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The processing configuration.
    """
    return ProcessingConfig(
        model=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        classes=tuple(args.classes),
        tracker=args.tracker,
        blur_faces=args.blur_faces,
        face_model=args.face_model,
        display=not args.no_display,
        cooldown_frames=args.cooldown_frames,
        cooldown_px=args.cooldown_px,
        frame_step=args.frame_step,
        layout=args.layout,
        save_tracks=args.save_tracks,
        replay_tracks=args.replay_tracks,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``line-counter`` console script.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = config_from_args(args)

    if args.gui or not args.input:
        from line_counter.gui import run_gui

        ignored = [opt for opt, val in (("--input", args.input), ("--output", args.output),
                                        ("--lines", args.lines), ("--save-lines", args.save_lines))
                   if val]
        if ignored:
            logger.warning("Ignored in GUI mode: %s", ", ".join(ignored))

        run_gui(config)
        return 0

    from line_counter.lines_io import load_lines, save_lines
    from line_counter.pipeline import default_output_path, process_video

    if args.lines:
        try:
            lines = load_lines(args.lines)
        except (OSError, ValueError) as exc:  # missing file, invalid JSON or bad entries
            logger.error("Cannot load lines from %s: %s", args.lines, exc)
            return 1
    elif args.no_display:
        parser.error("--no-display needs --lines (lines cannot be drawn without a window)")
    else:
        from line_counter.line_editor import draw_lines, read_first_frame

        frame = read_first_frame(args.input)
        if frame is None:
            logger.error("Could not read the first frame of %s", args.input)
            return 1
        lines = draw_lines(frame)
        if args.save_lines and lines:
            save_lines(lines, args.save_lines)
            logger.info("Saved %d line(s) to %s", len(lines), args.save_lines)

    if not lines:
        logger.error("No counting lines defined; nothing to do.")
        return 1

    if args.save_tracks and args.replay_tracks:
        parser.error("--save-tracks and --replay-tracks cannot be used together")

    output = args.output or default_output_path(args.input)
    counter = process_video(args.input, output, lines, config)
    if args.save_counts:
        from line_counter.counts_io import build_report, save_report

        save_report(args.save_counts, build_report(args.input, lines, counter.summary(), config))
        logger.info("Saved counts to %s", args.save_counts)
    return 0
