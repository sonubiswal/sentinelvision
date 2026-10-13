"""Benchmark runner for SentinelVision.

This script computes the held-out false-positive rate and violation F1 the
README advertises.  **The numbers are TBD until Phase 8**, when the SH17 and
Pictor-v3 labels exist -- nothing here fabricates a metric.  With no
``--labels`` the script refuses to print a score and exits 2.

    # metric math only, no data needed
    python benchmarks/evaluate_video.py --self-test

    # real run (needs a video and hand-labelled violations)
    python benchmarks/evaluate_video.py --video data/samples/shift.mp4 \
        --labels data/samples/shift.json --out benchmarks/results.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinelvision.events import ViolationEvent

#: Placeholder used everywhere a real measurement is still missing.
TBD = "TBD"


def safe_divide(numerator: float, denominator: float) -> float:
    """Division that yields ``0.0`` instead of raising on a zero denominator."""
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


@dataclass(frozen=True)
class FrameLabel:
    """Ground truth for one frame."""

    frame_index: int
    violations: frozenset[str]
    track_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameLabel:
        return cls(
            frame_index=int(data["frame_index"]),
            violations=frozenset(data.get("violations") or ()),
            track_id=None if data.get("track_id") is None else int(data["track_id"]),
        )


@dataclass(frozen=True)
class Score:
    """Detection score for a run."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        return safe_divide(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        return safe_divide(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return safe_divide(2 * precision * recall, precision + recall)

    def false_positive_rate(self, frames: int) -> float:
        """False positives *per processed frame*.

        A stricter rate (FP / negative opportunities) needs negative-frame
        labels, which arrive with the SH17 / Pictor-v3 sets in Phase 8.
        """
        return safe_divide(self.false_positives, max(int(frames), 1))

    def as_dict(self, frames: int | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }
        if frames is not None:
            data["frames"] = int(frames)
            data["false_positive_rate"] = round(self.false_positive_rate(frames), 4)
        return data


def load_labels(path: str | Path) -> list[FrameLabel]:
    """Load ``[{"frame_index": 0, "violations": [...]}, ...]`` from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("labels") or payload.get("frames") or []
    if not isinstance(payload, list):
        raise ValueError("labels file must contain a JSON list of frame labels")
    return [FrameLabel.from_dict(entry) for entry in payload]


def score_events(
    events: Sequence[ViolationEvent],
    labels: Sequence[FrameLabel],
    *,
    window: int = 3,
) -> Score:
    """Match events to labels within ``window`` frames.

    An event is a true positive when an unconsumed label within ``window``
    frames shares at least one violation class -- and shares the track ID when
    the label names one.  Unmatched events are false positives; unmatched
    labels are false negatives.
    """
    by_frame: dict[int, list[FrameLabel]] = defaultdict(list)
    for label in labels:
        by_frame[label.frame_index].append(label)

    consumed: set[tuple[int, int]] = set()
    true_positives = 0
    false_positives = 0

    for event in events:
        match: tuple[int, int] | None = None
        for offset in range(window + 1):
            frames = (
                [event.frame_index]
                if offset == 0
                else [event.frame_index - offset, event.frame_index + offset]
            )
            for candidate_frame in frames:
                for index, label in enumerate(by_frame.get(candidate_frame, ())):
                    key = (candidate_frame, index)
                    if key in consumed:
                        continue
                    if label.track_id is not None and label.track_id != event.track_id:
                        continue
                    if not (label.violations & event.violations):
                        continue
                    match = key
                    break
                if match is not None:
                    break
            if match is not None:
                break

        if match is None:
            false_positives += 1
        else:
            consumed.add(match)
            true_positives += 1

    false_negatives = sum(
        1
        for frame_index, entries in by_frame.items()
        for index in range(len(entries))
        if (frame_index, index) not in consumed
    )

    return Score(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def evaluate_video(
    video: str | Path,
    labels: Sequence[FrameLabel],
    *,
    weights: str = "yolov8n.pt",
    window: int = 3,
    max_frames: int = 0,
) -> dict[str, Any]:
    """Run the real pipeline over a video and score it."""
    import cv2

    from sentinelvision.detector import Detector
    from sentinelvision.pipeline import FramePipeline

    pipeline = FramePipeline(Detector.load(weights), camera=str(video))
    capture = cv2.VideoCapture(str(video))
    events: list[ViolationEvent] = []
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            events.extend(pipeline.step(frame).events)
            frames += 1
            if max_frames and frames >= max_frames:
                break
    finally:
        capture.release()

    score = score_events(events, labels, window=window)
    return {
        "video": str(video),
        "events": len(events),
        **score.as_dict(frames=frames),
        "vote_config": pipeline.vote_config.as_dict(),
    }


SELF_TEST_CASES: tuple[tuple[int, int, int], ...] = ((8, 2, 2), (0, 0, 0), (5, 0, 0), (0, 3, 3))


def self_test() -> dict[str, Any]:
    """Verify the metric math on constructed counts (not a benchmark result)."""
    report: dict[str, Any] = {}
    for tp, fp, fn in SELF_TEST_CASES:
        score = Score(tp, fp, fn)
        report[f"tp{tp}_fp{fp}_fn{fn}"] = score.as_dict()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_video",
        description="Score SentinelVision violation events against hand labels.",
    )
    parser.add_argument("--video", default=None, help="Video file to evaluate.")
    parser.add_argument("--labels", default=None, help="JSON frame labels.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Weights path or model name.")
    parser.add_argument("--window", type=int, default=3, help="Frame matching window (default: 3).")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all).")
    parser.add_argument("--out", default="benchmarks/results.json", help="Where to write results.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check the metric math on constructed counts and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0

    if not args.video or not args.labels:
        print(
            "No held-out video + labels supplied, so the person-attributed\n"
            "violation F1 and FP rate remain: " + TBD + "\n"
            "\n"
            "README rows stay at TBD until Phase 8 measures them on SH17 / Pictor-v3.\n"
            "Run with --self-test to verify the metric math."
        )
        return 2

    labels = load_labels(args.labels)
    results = evaluate_video(
        args.video,
        labels,
        weights=args.weights,
        window=args.window,
        max_frames=args.max_frames,
    )
    results["dataset"] = {"video": str(args.video), "labels": str(args.labels)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print(f"events={results['events']} frames={results['frames']}")
    print(f"precision={results['precision']} recall={results['recall']} f1={results['f1']}")
    print(f"false_positive_rate={results['false_positive_rate']} (per frame)")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
