"""WorkerState and WorkerRegistry: per-worker memory and its lifetime."""

from __future__ import annotations

import pytest

from sentinelvision.association import PPE_SLOTS, SlotState, associate
from sentinelvision.classes import normalize_class_name
from sentinelvision.detector import Detection
from sentinelvision.tracking import Box
from sentinelvision.worker import SLOT_ATTRIBUTES, WorkerRegistry, WorkerState


def det(
    class_name: str,
    xyxy: tuple[float, float, float, float],
    confidence: float = 0.9,
    track_id: int | None = 1,
) -> Detection:
    return Detection(
        class_id=0,
        class_name=normalize_class_name(class_name),  # Detector normalizes too
        raw_name=class_name,
        confidence=confidence,
        xyxy=xyxy,
        track_id=track_id,
    )


PERSON = (0.0, 0.0, 100.0, 200.0)


def association(
    detections: list[Detection],
    *,
    person_box: tuple[float, float, float, float] = PERSON,
    track_id: int = 1,
):
    """Build a PersonAssociation via the real associator (no hand-built fixtures)."""
    report = associate([det("person", person_box, track_id=track_id)], detections)
    return report.associations[0]


# --- WorkerState -----------------------------------------------------------
def test_worker_defaults_applied_in_post_init() -> None:
    worker = WorkerState(track_id=1, person_box=Box(0, 0, 10, 20))
    assert worker.history.maxlen == 10
    assert len(worker.history) == 0
    assert worker.violations == set()
    assert worker.frames_seen == 0
    assert worker.missing_frames == 0
    assert worker.first_seen_frame == -1
    assert worker.slot_states == {slot: SlotState.UNKNOWN for slot in PPE_SLOTS}
    assert worker.hardhat is None and worker.safety_vest is None and worker.mask is None


def test_slot_attributes_cover_every_configured_slot() -> None:
    assert set(SLOT_ATTRIBUTES) == set(PPE_SLOTS)


def test_slot_access_accepts_any_spelling() -> None:
    hardhat = det("Hardhat", (40.0, 10.0, 70.0, 40.0))
    worker = WorkerState(track_id=1, person_box=Box(0, 0, 100, 200))
    worker.set_slot("Hardhat", hardhat)
    assert worker.slot("hardhat") is hardhat
    assert worker.slot("HARDHAT") is hardhat
    assert worker.slot("safety_vest") is None


def test_unknown_slot_raises_rather_than_returning_none() -> None:
    worker = WorkerState(track_id=1, person_box=Box(0, 0, 10, 20))
    with pytest.raises(ValueError, match="unknown PPE slot"):
        worker.slot("gloves")
    with pytest.raises(ValueError, match="unknown PPE slot"):
        worker.set_slot("gloves", None)


def test_observe_updates_box_and_bookkeeping() -> None:
    worker = WorkerState(track_id=1, person_box=Box(0, 0, 1, 1))
    worker.observe(association([]), frame_index=4)
    assert worker.frames_seen == 1
    assert worker.first_seen_frame == 4
    assert worker.last_seen_frame == 4
    assert worker.missing_frames == 0
    assert worker.person_box.as_list() == list(PERSON)


def test_observe_records_slot_detections_and_states() -> None:
    worker = WorkerState(track_id=1, person_box=Box(0, 0, 1, 1))
    worker.observe(
        association([det("NO-Hardhat", (40.0, 10.0, 70.0, 40.0))]), frame_index=0
    )
    assert worker.state("hardhat") is SlotState.VIOLATION
    assert worker.hardhat is not None
    assert worker.hardhat.class_name == "no-hardhat"
    assert worker.violation_slots == frozenset({"hardhat"})
    assert worker.present_slots == frozenset()
    assert worker.unknown_slots == frozenset({"safety-vest", "mask"})


def test_observe_unknown_clears_stale_evidence() -> None:
    """Occlusion must not leave last frame's 'compliant' chip on the worker."""
    worker = WorkerState(track_id=1, person_box=Box(0, 0, 1, 1))
    worker.observe(association([det("Hardhat", (40.0, 10.0, 70.0, 40.0))]), frame_index=0)
    assert worker.slot("hardhat") is not None

    worker.observe(association([]), frame_index=1)
    assert worker.slot("hardhat") is None
    assert worker.state("hardhat") is SlotState.UNKNOWN


def test_frame_violations_is_derived_from_slot_states() -> None:
    worker = WorkerState(track_id=1, person_box=Box(0, 0, 1, 1))
    worker.observe(
        association(
            [
                det("NO-Hardhat", (40.0, 10.0, 70.0, 40.0)),
                det("NO-Safety Vest", (30.0, 60.0, 90.0, 130.0)),
            ]
        ),
        frame_index=0,
    )
    assert worker.frame_violations == {"no-hardhat", "no-safety-vest"}


def test_record_frame_bounds_the_history() -> None:
    worker = WorkerState(track_id=1, person_box=Box(0, 0, 1, 1), window=3)
    for _ in range(5):
        worker.record_frame({"no-hardhat"})
    assert len(worker.history) == 3
    assert worker.observations == 3


def test_worker_as_dict_shape() -> None:
    worker = WorkerState(track_id=7, person_box=Box(0, 0, 10, 20))
    worker.observe(association([], track_id=7), frame_index=0)
    data = worker.as_dict()
    assert data["track_id"] == 7
    assert data["box"] == [0.0, 0.0, 100.0, 200.0]
    assert set(data["slots"]) == set(PPE_SLOTS)
    assert data["slots"]["hardhat"]["state"] == "unknown"
    assert data["slots"]["hardhat"]["class"] is None
    assert data["violations"] == []
    assert data["frames_seen"] == 1


# --- WorkerRegistry --------------------------------------------------------
def test_touch_creates_once_and_then_updates() -> None:
    registry = WorkerRegistry()
    first = registry.touch(1, Box(0, 0, 10, 20), frame_index=0)
    second = registry.touch(1, Box(5, 5, 15, 25), frame_index=3)
    assert first is second
    assert len(registry) == 1
    assert second.person_box.as_list() == [5.0, 5.0, 15.0, 25.0]
    assert second.first_seen_frame == 0


def test_registry_container_protocol() -> None:
    registry = WorkerRegistry()
    registry.touch(2, Box(0, 0, 1, 1), frame_index=0)
    registry.touch(1, Box(0, 0, 1, 1), frame_index=0)
    assert 1 in registry
    assert 99 not in registry
    assert registry.track_ids == [1, 2]  # sorted
    assert [w.track_id for w in registry] == [1, 2]
    assert registry[1].track_id == 1
    assert registry.get(99) is None
    assert len(registry.workers) == 2


def test_registry_drop_and_clear() -> None:
    registry = WorkerRegistry()
    registry.touch(1, Box(0, 0, 1, 1), frame_index=0)
    registry.touch(2, Box(0, 0, 1, 1), frame_index=0)
    registry.drop(1)
    assert registry.track_ids == [2]
    registry.drop(1)  # idempotent
    registry.clear()
    assert len(registry) == 0


def test_registry_evicts_only_after_max_missing_frames() -> None:
    registry = WorkerRegistry(max_missing_frames=2, window=5)
    registry.touch(1, Box(0, 0, 1, 1), frame_index=0)
    assert registry[1].history.maxlen == 5

    assert registry.evict_stale(1) == []  # missing = 1
    assert registry.evict_stale(2) == []  # missing = 2 (not > 2)
    assert registry.evict_stale(3) == [1]  # missing = 3 > 2
    assert registry.track_ids == []


def test_registry_does_not_age_a_worker_seen_this_frame() -> None:
    registry = WorkerRegistry(max_missing_frames=0)
    registry.touch(1, Box(0, 0, 1, 1), frame_index=5)
    assert registry.evict_stale(5) == []
    assert registry.track_ids == [1]


def test_registry_as_dict_is_keyed_and_sorted() -> None:
    registry = WorkerRegistry()
    registry.touch(2, Box(0, 0, 1, 1), frame_index=0)
    registry.touch(1, Box(0, 0, 1, 1), frame_index=0)
    data = registry.as_dict()
    assert list(data) == [1, 2]
    assert data[1]["track_id"] == 1
