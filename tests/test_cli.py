"""CLI tests -- model, capture and detector are all mocked (no weights, no camera)."""

from __future__ import annotations

import numpy as np
import pytest

from sentinelvision import cli
from sentinelvision.classes import VIOLATION_CLASSES
from sentinelvision.config import CLASS_THRESHOLDS
from sentinelvision.detector import Detector
from tests.fakes import FakeBoxes, FakeModel

NAMES = {0: "person", 1: "NO-Hardhat", 2: "Hardhat"}


class FakeCapture:
    """Yields ``frames`` readable frames, then reports end-of-stream."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self.released = False
        self.read_count = 0

    def read(self):
        self.read_count += 1
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


def fake_detector() -> Detector:
    model = FakeModel(
        FakeBoxes(
            xyxy=[[0, 0, 10, 20], [5, 5, 15, 25], [8, 8, 18, 28]],
            conf=[0.90, 0.80, 0.70],
            cls=[0, 1, 2],
            ids=[1, 2, 3],
        ),
        names=NAMES,
    )
    return Detector(model)


@pytest.fixture
def patched(monkeypatch):
    """Patch the CLI seams: detector loading and capture opening."""
    captures: list[FakeCapture] = []

    def _open(source):
        cap = FakeCapture([np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)])
        captures.append(cap)
        return cap

    monkeypatch.setattr(cli, "_load_detector", lambda settings: fake_detector())
    monkeypatch.setattr(cli, "_open_capture", _open)
    return captures


# --- parser ---------------------------------------------------------------
def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args([])
    assert args.source == "0"
    assert args.model is None
    assert args.conf is None
    assert args.show is False
    assert args.list_classes is False
    assert args.max_frames is None


def test_parser_accepts_source_and_flags() -> None:
    args = cli.build_parser().parse_args(
        ["--source", "rtsp://cam/stream", "--conf", "0.4", "--show", "--max-frames", "5"]
    )
    assert args.source == "rtsp://cam/stream"
    assert args.conf == pytest.approx(0.4)
    assert args.show is True
    assert args.max_frames == 5


# --- list-classes ---------------------------------------------------------
def test_list_classes_prints_every_class_and_exits_zero(capsys) -> None:
    assert cli.main(["--list-classes"]) == 0
    out = capsys.readouterr().out
    for name in CLASS_THRESHOLDS:
        assert name in out
    assert out.count("\n") >= len(CLASS_THRESHOLDS)
    assert f"{len(CLASS_THRESHOLDS)} classes" in out
    assert f"{len(VIOLATION_CLASSES)} violations" in out


def test_list_classes_never_touches_the_model(monkeypatch, capsys) -> None:
    def _explode(settings):  # pragma: no cover - must never run
        raise AssertionError("detector must not load for --list-classes")

    monkeypatch.setattr(cli, "_load_detector", _explode)
    assert cli.main(["--list-classes"]) == 0
    assert capsys.readouterr().out != ""


# --- run loop -------------------------------------------------------------
def test_run_loop_counts_frames_and_violations(patched, capsys) -> None:
    assert cli.main(["--source", "0", "--max-frames", "2"]) == 0
    out = capsys.readouterr().out
    assert "Processed 2 frame(s)" in out
    # One NO-Hardhat per frame over two frames.
    assert "2 violation detection(s)" in out
    assert patched[0].released is True


def test_run_loop_stops_at_end_of_stream(patched, capsys) -> None:
    assert cli.main(["--source", "0"]) == 0
    assert "Processed 3 frame(s)" in capsys.readouterr().out
    assert patched[0].read_count == 4


def test_max_frames_zero_means_unlimited(patched, capsys) -> None:
    assert cli.main(["--source", "0", "--max-frames", "0"]) == 0
    assert "Processed 3 frame(s)" in capsys.readouterr().out


def test_run_loop_always_releases_the_capture(patched) -> None:
    cli.main(["--source", "0", "--max-frames", "1"])
    assert patched[0].released is True


def test_run_loop_never_calls_imshow_without_show(patched, monkeypatch) -> None:
    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("imshow must not be called without --show")

    monkeypatch.setattr("cv2.imshow", _explode)
    assert cli.main(["--source", "0", "--max-frames", "1"]) == 0


class FakeWriter:
    """Stand-in for ``cv2.VideoWriter`` -- records frames instead of encoding."""

    instances: list[FakeWriter] = []

    def __init__(self, *args, **kwargs) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False
        FakeWriter.instances.append(self)

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


def test_show_draws_the_frame_then_quits_on_q(patched, monkeypatch, capsys) -> None:
    seen = {"imshow": 0, "destroy": 0}

    def _imshow(name, frame):
        seen["imshow"] += 1
        assert frame.sum() > 0  # annotated in place before display

    monkeypatch.setattr("cv2.imshow", _imshow)
    monkeypatch.setattr("cv2.waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(
        "cv2.destroyAllWindows", lambda: seen.__setitem__("destroy", seen["destroy"] + 1)
    )

    assert cli.main(["--source", "0", "--show"]) == 0
    assert seen["imshow"] == 1
    assert seen["destroy"] == 1
    assert "Processed 1 frame(s)" in capsys.readouterr().out


def test_save_writes_every_annotated_frame(patched, monkeypatch, tmp_path) -> None:
    FakeWriter.instances.clear()
    monkeypatch.setattr("cv2.VideoWriter", FakeWriter)

    assert cli.main(["--source", "0", "--save", str(tmp_path / "annotated.mp4")]) == 0
    assert len(FakeWriter.instances) == 1
    writer = FakeWriter.instances[0]
    assert writer.released is True
    assert len(writer.frames) == 3
    assert all(frame.sum() > 0 for frame in writer.frames)
