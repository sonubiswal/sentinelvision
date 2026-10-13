"""Command-line entry point: ``python -m sentinelvision`` / ``sentinelvision``.

Phase 2 wires the CLI to :class:`~sentinelvision.pipeline.FramePipeline` rather
than the bare detector, so the annotated preview already shows track IDs and
committed violations have passed the N-of-M vote.  The detector and the capture
remain injectable seams, which is why the CLI tests need no weights and no
camera.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from typing import Any, TextIO

from .classes import VIOLATION_CLASSES
from .config import CLASS_THRESHOLDS, Settings
from .detector import Detector
from .events import EventBus
from .logging_setup import configure_logging, get_logger
from .pipeline import FramePipeline
from .visualize import draw_detections

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinelvision",
        description="SentinelVision -- per-worker PPE compliance from commodity cameras.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index, RTSP URL or video file path (default: 0).",
    )
    parser.add_argument(
        "--model", default=None, help="Weights path or model name (default: yolov8n.pt)."
    )
    parser.add_argument(
        "--device", default=None, help="Inference device, e.g. cpu, cuda:0 (default: cpu)."
    )
    parser.add_argument("--conf", type=float, default=None, help="Override the global query threshold.")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold.")
    parser.add_argument("--show", action="store_true", help="Display an annotated preview window.")
    parser.add_argument("--save", default=None, help="Write the annotated video to this path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (0 = unlimited).")
    parser.add_argument("--list-classes", action="store_true", help="Print the class taxonomy and exit.")
    parser.add_argument("--json-logs", action="store_true", help="Emit JSON logs instead of console logs.")

    # Phase 2 -- pipeline controls.
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable tracking (debug only: without track IDs no violation can persist).",
    )
    parser.add_argument(
        "--vote-window",
        type=int,
        default=None,
        help=f"Temporal voting window N (default: {Settings().vote_window}).",
    )
    parser.add_argument(
        "--vote-threshold",
        type=int,
        default=None,
        help=f"Votes required to commit K (default: {Settings().vote_threshold}).",
    )
    parser.add_argument(
        "--clear-hysteresis",
        type=int,
        default=None,
        help=f"Consecutive clean frames needed to resolve (default: {Settings().clear_hysteresis}).",
    )
    parser.add_argument(
        "--events-jsonl",
        default=None,
        help="Append committed violations as JSON Lines to this path.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print pipeline counters as JSON at the end of the run.",
    )
    return parser


def _open_event_sink(path: str) -> TextIO:
    """Open the JSONL event sink.

    Ownership transfers to the caller: the run loop registers the handle with an
    ``ExitStack``, so it is closed exactly once on every exit path.
    """
    return open(path, "w", encoding="utf-8")  # noqa: SIM115


def _open_capture(source: Any) -> Any:
    """Seam for tests: open a cv2 capture for a webcam index or a path."""
    import cv2

    text = str(source)
    if text.isdigit():
        return cv2.VideoCapture(int(text))
    return cv2.VideoCapture(text)


def _load_detector(settings: Settings) -> Detector:
    """Seam for tests: build a Detector from settings."""
    return Detector.load(settings.weights, device=settings.device, tracker=settings.tracker)


def _build_pipeline(settings: Settings, *, camera: str | None = None) -> FramePipeline:
    """Seam for tests: wire a pipeline around the detector."""
    return FramePipeline(_load_detector(settings), settings=settings, camera=camera, bus=EventBus())


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Fold CLI flags into settings.  Config stays additive: no Phase 1 key moves."""
    if args.model is not None:
        settings.weights = args.model
    if args.device is not None:
        settings.device = args.device
    if args.iou is not None:
        settings.iou = args.iou
    if args.no_track:
        settings.track = False
    if args.vote_window is not None:
        settings.vote_window = args.vote_window
    if args.vote_threshold is not None:
        settings.vote_threshold = args.vote_threshold
    if args.clear_hysteresis is not None:
        settings.clear_hysteresis = args.clear_hysteresis
    return settings


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _apply_overrides(Settings(), args)
    configure_logging(settings.log_level, json_logs=args.json_logs or settings.json_logs)
    log = get_logger("sentinelvision.cli")

    if args.list_classes:
        for name, threshold in CLASS_THRESHOLDS.items():
            print(f"{name}\t{threshold:.2f}")
        print(
            f"\n{len(CLASS_THRESHOLDS)} classes, {len(VIOLATION_CLASSES)} violations: "
            f"{', '.join(VIOLATION_CLASSES)}"
        )
        return 0

    max_frames = args.max_frames if args.max_frames is not None else settings.max_frames

    # Settings already carry every CLI override (see _apply_overrides), and
    # FramePipeline derives its voter and registry windows from them in
    # __init__.  Poking pipeline.vote_config afterwards would NOT reach the
    # already-constructed voter, so it is deliberately not done here.
    pipeline = _build_pipeline(settings, camera=str(args.source))
    capture = _open_capture(args.source)

    # ExitStack keeps the optional JSONL sink inside a context manager without
    # an if/else ladder wrapped around the whole run loop.
    stack = ExitStack()
    writer = None
    events_file = (
        stack.enter_context(_open_event_sink(args.events_jsonl))
        if args.events_jsonl
        else None
    )
    frames = 0
    violation_detections = 0
    event_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = pipeline.step(frame)
            frames += 1
            violation_detections += sum(1 for d in result.detections if d.is_violation)
            event_count += len(result.events)
            if events_file is not None:
                for event in result.events:
                    events_file.write(event.to_json() + "\n")

            if args.show or args.save:
                draw_detections(frame, result.detections)

            if args.save:
                if writer is None:
                    import cv2

                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        args.save, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height)
                    )
                writer.write(frame)

            if args.show:
                import cv2

                cv2.imshow("SentinelVision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames and frames >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        stack.close()
        if args.show:
            import cv2

            cv2.destroyAllWindows()

    log.info(
        "run_finished",
        frames=frames,
        violation_detections=violation_detections,
        committed_events=event_count,
    )
    print(
        f"Processed {frames} frame(s); {violation_detections} violation detection(s); "
        f"{event_count} committed event(s)."
    )
    if args.stats:
        print(json.dumps(pipeline.stats.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
