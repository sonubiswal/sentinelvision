"""FramePipeline end-to-end: detect → associate → policy → vote → events.

Every test drives the real pipeline against a scripted fake model, so no
weights, no torch and no camera are involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinelvision.association import SlotState
from sentinelvision.config import Settings
from sentinelvision.detector import Detector
from sentinelvision.events import EventBus
from sentinelvision.pipeline import (
    DEFAULTS_IN_USE,
    FramePipeline,
    GlobalPolicy,
    PipelineStats,
)
from sentinelvision.tracking import Box
from sentinelvision.voting import VoteConfig
from tests.fakes import FakeBoxes, ScriptedModel, make_boxes

NAMES = {
    0: "person",
    1: "NO-Hardhat",
    2: "Hardhat",
    3: "Safety Vest",
    4: "NO-Safety Vest",
}

PERSON_BOX = [0.0, 0.0, 100.0, 200.0]
HARDHAT_BOX = [40.0, 10.0, 70.0, 40.0]

# window 4 / threshold 3 keeps the loops short while exercising the same code
# path as the shipped 10/6 defaults (asserted separately below).
FAST = VoteConfig(window=4, threshold=3, clear_hysteresis=2)


def violating() -> FakeBoxes:
    return make_boxes([(PERSON_BOX, 0.90, 0), (HARDHAT_BOX, 0.80, 1)], ids=[1, 1])


def compliant() -> FakeBoxes:
    return make_boxes([(PERSON_BOX, 0.90, 0), (HARDHAT_BOX, 0.75, 2)], ids=[1, 1])


def clean() -> FakeBoxes:
    return make_boxes([(PERSON_BOX, 0.90, 0)], ids=[1])


def empty() -> FakeBoxes:
    return make_boxes([])


def strip_ids(boxes: FakeBoxes) -> FakeBoxes:
    """The same boxes with no track IDs, for the ``track=False`` path."""
    return FakeBoxes(
        xyxy=boxes.xyxy.tolist(),
        conf=boxes.conf.tolist(),
        cls=[int(value) for value in boxes.cls.tolist()],
    )


def frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def build(script: list, *, ids: bool = True, vote_config: VoteConfig = FAST, **kwargs):
    if not ids:
        script = [strip_ids(boxes) for boxes in script]
    model = ScriptedModel(script, names=NAMES)
    pipeline = FramePipeline(
        Detector(model),
        settings=Settings(),
        vote_config=vote_config,
        camera="cam-01",
        **kwargs,
    )
    return pipeline, model


# --- guards ----------------------------------------------------------------
def test_step_requires_a_detector() -> None:
    with pytest.raises(RuntimeError, match="no detector"):
        FramePipeline(None).step(frame())


def test_step_rejects_a_none_frame() -> None:
    pipeline, _ = build([clean()])
    with pytest.raises(ValueError, match="frame must be"):
        pipeline.step(None)  # type: ignore[arg-type]


def test_defaults_in_use_match_the_shipped_config() -> None:
    assert DEFAULTS_IN_USE["vote_window"] == 10
    assert DEFAULTS_IN_USE["vote_threshold"] == 6
    assert DEFAULTS_IN_USE["clear_hysteresis"] == 3
    assert DEFAULTS_IN_USE["association_iou_min"] == 0.30
    assert DEFAULTS_IN_USE["association_containment_min"] == 0.40


# --- the core behaviour ----------------------------------------------------
def test_violation_commits_only_after_the_vote() -> None:
    """3 consecutive violating frames under K=3: nothing, then one event."""
    pipeline, model = build([violating()] * 3)

    first = pipeline.step(frame())
    second = pipeline.step(frame())
    third = pipeline.step(frame())

    assert first.events == () and second.events == ()
    assert len(third.events) == 1

    evt = third.events[0]
    assert evt.track_id == 1
    assert evt.violations == frozenset({"no-hardhat"})
    assert evt.frame_index == 2
    assert evt.zone is None
    assert evt.camera == "cam-01"
    assert evt.confidence == pytest.approx(0.80)
    assert evt.is_intact() is True

    assert len(model.track_calls) == 3
    assert model.track_calls[0]["tracker"] == "bytetrack.yaml"
    assert model.track_calls[0]["persist"] is True
    assert model.track_calls[0]["verbose"] is False


def test_frame_level_violation_is_visible_before_it_commits() -> None:
    pipeline, _ = build([violating()])
    result = pipeline.step(frame())
    assert result.violations == frozenset({"no-hardhat"})
    assert result.events == ()
    assert result.active_violations == frozenset()
    assert result.association.violations == frozenset({"no-hardhat"})


def test_violation_clears_after_the_hysteresis() -> None:
    pipeline, _ = build([violating()] * 3 + [clean()] * 3)

    for _ in range(3):
        pipeline.step(frame())
    assert pipeline.active_violations[1] == frozenset({"no-hardhat"})

    first_clean = pipeline.step(frame())
    assert first_clean.events == ()
    assert pipeline.active_violations[1] == frozenset({"no-hardhat"})

    second_clean = pipeline.step(frame())
    assert second_clean.votes[0].cleared == frozenset({"no-hardhat"})
    assert pipeline.active_violations[1] == frozenset()


def test_occlusion_is_unknown_and_never_a_violation() -> None:
    """A worker who is simply not visible must never fault."""
    pipeline, _ = build([clean()] * 8)
    results = pipeline.run([frame() for _ in range(8)])

    worker = pipeline.worker(1)
    assert worker is not None
    assert worker.state("hardhat") is SlotState.UNKNOWN
    assert worker.slot("hardhat") is None
    assert worker.violations == set()
    assert all(result.events == () for result in results)
    assert pipeline.stats.committed == 0


def test_compliant_worker_never_accumulates_votes() -> None:
    pipeline, _ = build([compliant()] * 6)
    results = pipeline.run([frame() for _ in range(6)])

    worker = pipeline.worker(1)
    assert worker is not None
    assert worker.state("hardhat") is SlotState.PRESENT
    assert worker.violations == set()
    assert all(r.votes[0].votes == {} for r in results)


def test_violation_and_compliance_are_distinguished_per_slot() -> None:
    """One worker in one frame: compliant hardhat, missing vest, occluded mask."""
    script = [
        make_boxes(
            [
                (PERSON_BOX, 0.9, 0),
                (HARDHAT_BOX, 0.8, 2),  # compliant hardhat
                ([30.0, 60.0, 90.0, 130.0], 0.7, 4),  # NO-Safety Vest
            ],
            ids=[1, 1, 1],
        )
    ] * 3
    pipeline, _ = build(script)
    results = pipeline.run([frame() for _ in range(3)])
    result = results[-1]

    worker = pipeline.worker(1)
    assert worker is not None
    assert worker.state("hardhat") is SlotState.PRESENT
    assert worker.state("safety-vest") is SlotState.VIOLATION
    assert worker.state("mask") is SlotState.UNKNOWN
    assert worker.violations == {"no-safety-vest"}
    assert result.active_violations == frozenset({"no-safety-vest"})
    assert len(result.events) == 1


# --- policy seam (Phase 3 entry point) -------------------------------------
def test_policy_can_disable_a_requirement() -> None:
    """The zone seam: if a zone does not require hardhats, it never votes."""
    pipeline, _ = build(
        [violating()] * 6, policy=GlobalPolicy(required=frozenset({"safety-vest"}))
    )
    pipeline.run([frame() for _ in range(6)])

    worker = pipeline.worker(1)
    assert worker is not None
    assert worker.state("hardhat") is SlotState.VIOLATION  # still observed...
    assert worker.violations == set()  # ...but not required here
    assert pipeline.stats.committed == 0


def test_policy_provides_the_zone_label_on_events() -> None:
    pipeline, _ = build([violating()] * 3, policy=GlobalPolicy(zone_name="welding-bay"))
    events = pipeline.events_from(pipeline.run([frame() for _ in range(3)]))
    assert len(events) == 1
    assert events[0].zone == "welding-bay"


def test_global_policy_defaults_to_every_slot() -> None:
    policy = GlobalPolicy()
    required = policy.required_slots(Box(0, 0, 1, 1))
    assert required == frozenset({"hardhat", "safety-vest", "mask"})
    assert policy.zone_for(Box(0, 0, 1, 1)) is None


# --- identity --------------------------------------------------------------
def test_without_tracking_nothing_can_commit() -> None:
    """No cross-frame identity means no persistence, so voting correctly declines."""
    pipeline, model = build([violating()] * 4, ids=False, track=False)
    results = pipeline.run([frame() for _ in range(4)])

    assert all(result.events == () for result in results)
    assert pipeline.stats.committed == 0
    assert len(pipeline.registry) == 4  # a new synthetic ID per frame
    assert all(track_id < 0 for track_id in pipeline.registry.track_ids)
    assert model.track_calls == []
    assert len(model.calls) == 4


def test_track_ids_are_keys_on_the_worker_map() -> None:
    pipeline, _ = build([violating()] * 3)
    pipeline.run([frame() for _ in range(3)])
    assert pipeline.registry.track_ids == [1]
    assert pipeline.worker(1) is pipeline.registry[1]
    assert pipeline.worker(2) is None


def test_stale_workers_are_evicted() -> None:
    pipeline, _ = build([violating(), empty(), empty()], max_missing_frames=1)
    pipeline.run([frame() for _ in range(3)])
    assert pipeline.registry.track_ids == []
    assert pipeline.stats.workers_evicted == 1


# --- events + stats --------------------------------------------------------
def test_committed_events_are_published_to_the_bus() -> None:
    bus = EventBus()
    pipeline, _ = build([violating()] * 3, bus=bus)
    pipeline.run([frame() for _ in range(3)])

    assert bus.stats.published == 1
    assert len(bus.recent()) == 1
    assert bus.recent(1)[0].violations == frozenset({"no-hardhat"})


def test_no_publish_without_an_event() -> None:
    bus = EventBus()
    pipeline, _ = build([clean()] * 3, bus=bus)
    pipeline.run([frame() for _ in range(3)])
    assert bus.stats.published == 0


def test_stats_track_the_run() -> None:
    pipeline, _ = build([violating(), clean(), compliant()])
    pipeline.step(frame())

    assert pipeline.stats.frames == 1
    assert pipeline.stats.detections == 2
    assert pipeline.stats.persons == 1
    assert pipeline.stats.associations == 1
    assert pipeline.stats.bound_ppe == 1
    assert pipeline.stats.unassigned_ppe == 0
    assert pipeline.stats.workers_seen == 1

    pipeline.run([frame(), frame()])
    assert pipeline.stats.frames == 3
    assert pipeline.stats.detections == 5


def test_unassigned_ppe_is_counted_not_silently_dropped() -> None:
    script = [make_boxes([(PERSON_BOX, 0.9, 0), ([900.0, 900.0, 920.0, 920.0], 0.8, 1)])]
    pipeline, _ = build(script)
    pipeline.step(frame())
    assert pipeline.stats.bound_ppe == 0
    assert pipeline.stats.unassigned_ppe == 1


def test_reset_clears_all_state() -> None:
    pipeline, _ = build([violating()] * 3)
    pipeline.run([frame() for _ in range(3)])
    assert pipeline.frame_index == 2

    pipeline.reset()
    assert pipeline.frame_index == -1
    assert len(pipeline.registry) == 0
    assert pipeline.stats == PipelineStats()


def test_as_dict_exposes_config_and_workers() -> None:
    pipeline, _ = build([violating()] * 3)
    pipeline.run([frame() for _ in range(3)])
    data = pipeline.as_dict()

    assert data["track"] is True
    assert data["frame_index"] == 2
    assert data["policy"] == "GlobalPolicy"
    assert data["vote_config"]["window"] == 4
    assert data["stats"]["frames"] == 3
    assert data["workers"][1]["violations"] == ["no-hardhat"]


def test_frame_result_as_dict_serializes_events_and_workers() -> None:
    pipeline, _ = build([violating()] * 3)
    result = pipeline.run([frame() for _ in range(3)])[-1]
    data = result.as_dict()

    assert data["frame_index"] == 2
    assert data["detections"] == 2
    assert data["associations"] == 1
    assert data["events"][0]["violations"] == ["no-hardhat"]
    assert data["workers"][0]["track_id"] == 1
    assert len(result.persons) == 1
