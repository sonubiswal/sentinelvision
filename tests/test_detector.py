"""Detector tests: class-aware thresholds and result parsing."""

from __future__ import annotations

import numpy as np
import pytest

from sentinelvision.classes import normalize_class_name
from sentinelvision.config import CLASS_THRESHOLDS
from sentinelvision.detector import Detection, Detector
from tests.fakes import FakeBoxes, FakeModel

NAMES = {0: "person", 1: "NO-Hardhat", 2: "Safety Vest", 3: "NO-Safety Vest"}


def make_model(boxes: FakeBoxes) -> FakeModel:
    return FakeModel(boxes, names=NAMES)


def test_detector_without_model_raises() -> None:
    with pytest.raises(RuntimeError, match="no model"):
        Detector().detect(np.zeros((8, 8, 3), dtype=np.uint8))


def test_detect_with_none_frame_raises() -> None:
    detector = Detector(make_model(FakeBoxes([], [], [])))
    with pytest.raises(ValueError, match="frame must be"):
        detector.detect(None)  # type: ignore[arg-type]


def test_model_is_queried_no_higher_than_the_lowest_configured_threshold() -> None:
    """Per-class filtering happens *after* the model call.

    So the query threshold must be <= every per-class threshold (the fallback
    included) -- otherwise a low-confidence class would be filtered by the
    model itself, before class-aware filtering could ever see it.
    """
    model = make_model(FakeBoxes([], [], []))
    detector = Detector(model)
    detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

    query = model.calls[0]["conf"]
    expected = min([*detector.thresholds.values(), detector.fallback_threshold])
    assert query == pytest.approx(expected)
    assert query <= min(CLASS_THRESHOLDS.values())


def test_class_aware_threshold_filters_a_marginal_violation() -> None:
    """NO-Hardhat needs 0.65; a 0.50 detection must not create a violation."""
    model = make_model(
        FakeBoxes(
            xyxy=[[0, 0, 10, 20], [5, 5, 15, 25]],
            conf=[0.90, 0.50],
            cls=[0, 1],
        )
    )
    detections = Detector(model).detect(np.zeros((32, 32, 3), dtype=np.uint8))
    assert [d.class_name for d in detections] == ["person"]


def test_high_confidence_violation_passes_and_is_flagged() -> None:
    model = make_model(
        FakeBoxes(xyxy=[[5, 5, 15, 25]], conf=[0.72], cls=[1])
    )
    detections = Detector(model).detect(np.zeros((32, 32, 3), dtype=np.uint8))
    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "no-hardhat"
    assert det.raw_name == "NO-Hardhat"
    assert det.is_violation is True
    assert det.confidence == pytest.approx(0.72)
    assert det.xyxy == (5.0, 5.0, 15.0, 25.0)


def test_mixed_batch_keeps_only_survivors_and_parses_headers() -> None:
    model = make_model(
        FakeBoxes(
            xyxy=[[0, 0, 10, 10], [2, 2, 12, 22], [4, 4, 14, 24], [6, 6, 16, 26]],
            conf=[0.91, 0.66, 0.40, 0.80],
            cls=[0, 1, 2, 3],
            ids=[101, 102, 103, 104],
        )
    )
    detections = Detector(model).detect(np.zeros((32, 32, 3), dtype=np.uint8))
    assert [d.class_name for d in detections] == ["person", "no-hardhat", "no-safety-vest"]
    assert [d.track_id for d in detections] == [101, 102, 104]
    assert detections[1].is_violation is True
    assert detections[2].is_violation is True


def test_names_can_be_a_list() -> None:
    class ListNamesModel(FakeModel):
        def __init__(self, boxes: FakeBoxes) -> None:
            super().__init__(boxes, names=["person", "NO-Mask"])

    model = ListNamesModel(FakeBoxes(xyxy=[[0, 0, 4, 4]], conf=[0.9], cls=[1]))
    detections = Detector(model).detect(np.zeros((16, 16, 3), dtype=np.uint8))
    assert detections[0].class_name == "no-mask"
    assert detections[0].is_violation is True


def test_unknown_class_uses_fallback_threshold() -> None:
    class SwappedModel(FakeModel):
        def __init__(self, boxes: FakeBoxes) -> None:
            super().__init__(boxes, names={0: "Gloves"})

    model = SwappedModel(FakeBoxes(xyxy=[[0, 0, 4, 4]], conf=[0.30], cls=[0]))
    detector = Detector(model, fallback_threshold=0.25)
    assert len(detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))) == 1
    detector_strict = Detector(model, fallback_threshold=0.50)
    assert detector_strict.detect(np.zeros((16, 16, 3), dtype=np.uint8)) == []


def test_explicit_conf_overrides_the_query_threshold_only() -> None:
    model = make_model(FakeBoxes(xyxy=[[0, 0, 4, 4]], conf=[0.55], cls=[1]))
    detector = Detector(model)
    assert detector.detect(np.zeros((16, 16, 3), dtype=np.uint8), conf=0.1) == []
    assert model.calls[0]["conf"] == pytest.approx(0.1)


def test_empty_boxes_is_not_an_error() -> None:
    model = make_model(FakeBoxes([], [], []))
    assert Detector(model).detect(np.zeros((16, 16, 3), dtype=np.uint8)) == []


def test_detection_geometry_properties() -> None:
    det = Detection(
        class_id=1,
        class_name=normalize_class_name("NO-Hardhat"),
        raw_name="NO-Hardhat",
        confidence=0.7,
        xyxy=(10.0, 20.0, 30.0, 60.0),
    )
    assert det.width == pytest.approx(20.0)
    assert det.height == pytest.approx(40.0)
    assert det.area == pytest.approx(800.0)
    assert det.center == pytest.approx((20.0, 40.0))
    assert det.is_violation is True


def test_load_without_ultralytics_gives_actionable_error() -> None:
    import importlib.util

    if importlib.util.find_spec("ultralytics") is not None:
        pytest.skip("ultralytics is installed in this environment")
    with pytest.raises(ImportError, match="ultralytics is required"):
        Detector.load("yolov8n.pt")


def test_result_without_boxes_is_skipped() -> None:
    model = FakeModel(None, names=NAMES)
    assert Detector(model).detect(np.zeros((16, 16, 3), dtype=np.uint8)) == []
