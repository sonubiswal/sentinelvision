"""FramePipeline: the orchestrator that turns frames into violation events.

    frame -> detect/track -> associate -> policy filter -> vote -> emit

Each arrow is a separate, independently tested module.  The pipeline itself
holds only frame counters and the seams (detector, policy, voter, bus, clock),
so a phase can swap any one of them without touching the others.

The ``Policy`` protocol is the Phase 3 seam.  Phase 2 ships ``GlobalPolicy``
(no zones: every worker must wear everything); Phase 3 drops in a polygon-geofence
implementation with per-zone PPE requirements and nothing else changes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .association import (
    PPE_SLOTS,
    AssociationReport,
    PersonAssociation,
    associate,
    ppe_slot,
    relocate_track_id,
)
from .config import (
    ASSOCIATION_CONTAINMENT_MIN,
    ASSOCIATION_IOU_MIN,
    VOTE_WINDOW,
    WORKER_MAX_MISSING_FRAMES,
    Settings,
)
from .detector import Detection, Detector
from .events import EventBus, ViolationEvent
from .tracking import (
    PERSON_CLASS,
    Box,
    SyntheticTrackIds,
    persons,
    with_track_id,
)
from .voting import DEFAULT_VOTE_CONFIG, TemporalVoter, VoteConfig, VoteResult
from .worker import WorkerRegistry, WorkerState


@runtime_checkable
class Policy(Protocol):
    """Decides which PPE slots a worker is required to wear.

    Phase 3 replaces this with per-zone geofence policy.
    """

    def required_slots(self, worker_box: Box) -> frozenset[str]:
        """PPE slots required at ``worker_box``'s location."""
        ...

    def zone_for(self, worker_box: Box) -> str | None:
        """Human-readable zone name at that location, if any."""
        ...


@dataclass(frozen=True)
class GlobalPolicy:
    """Phase 2 default: one site-wide requirement, no zones.

    ``required`` defaults to every configured PPE slot, so behaviour matches the
    taxonomy out of the box.
    """

    required: frozenset[str] = field(default_factory=lambda: frozenset(PPE_SLOTS))
    zone_name: str | None = None

    def required_slots(self, worker_box: Box) -> frozenset[str]:
        return self.required

    def zone_for(self, worker_box: Box) -> str | None:
        return self.zone_name


@dataclass
class PipelineStats:
    """Running counters for logging, dashboards and Prometheus."""

    frames: int = 0
    detections: int = 0
    persons: int = 0
    associations: int = 0
    bound_ppe: int = 0
    unassigned_ppe: int = 0
    unseen_persons: int = 0
    events: int = 0
    committed: int = 0
    cleared: int = 0
    workers_seen: int = 0
    workers_evicted: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "detections": self.detections,
            "persons": self.persons,
            "associations": self.associations,
            "bound_ppe": self.bound_ppe,
            "unassigned_ppe": self.unassigned_ppe,
            "unseen_persons": self.unseen_persons,
            "events": self.events,
            "committed": self.committed,
            "cleared": self.cleared,
            "workers_seen": self.workers_seen,
            "workers_evicted": self.workers_evicted,
        }


@dataclass
class FrameResult:
    """Everything one :meth:`FramePipeline.step` produced."""

    frame_index: int
    detections: tuple[Detection, ...]
    association: AssociationReport
    votes: tuple[VoteResult, ...]
    events: tuple[ViolationEvent, ...]
    workers: tuple[WorkerState, ...]

    @property
    def persons(self) -> tuple[Detection, ...]:
        return tuple(a.person for a in self.association.associations)

    @property
    def violations(self) -> frozenset[str]:
        """Violation classes observed this frame (pre-vote)."""
        return self.association.violations

    @property
    def active_violations(self) -> frozenset[str]:
        """Violation classes currently committed across all workers."""
        active: set[str] = set()
        for result in self.votes:
            active |= result.active
        return frozenset(active)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "detections": len(self.detections),
            "associations": len(self.association.associations),
            "unassigned_ppe": len(self.association.unassigned),
            "events": [event.to_dict() for event in self.events],
            "workers": [worker.as_dict() for worker in self.workers],
        }


class FramePipeline:
    """Turns a stream of frames into committed, track-bound violation events."""

    def __init__(
        self,
        detector: Detector | None = None,
        *,
        settings: Settings | None = None,
        policy: Policy | None = None,
        vote_config: VoteConfig | None = None,
        bus: EventBus | None = None,
        track: bool | None = None,
        camera: str | None = None,
        clock: Callable[[], float] = time.time,
        association_iou_min: float | None = None,
        association_containment_min: float | None = None,
        max_missing_frames: int | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.detector = detector
        self.policy: Policy = policy or GlobalPolicy()
        self.vote_config = vote_config or VoteConfig(
            window=self.settings.vote_window,
            threshold=self.settings.vote_threshold,
            clear_hysteresis=self.settings.clear_hysteresis,
        )
        self.bus = bus
        self.track = self.settings.track if track is None else bool(track)
        self.camera = camera
        self.clock = clock
        self.iou_min = (
            self.settings.association_iou_min
            if association_iou_min is None
            else float(association_iou_min)
        )
        self.containment_min = (
            self.settings.association_containment_min
            if association_containment_min is None
            else float(association_containment_min)
        )

        self.voter = TemporalVoter(self.vote_config)
        self.registry = WorkerRegistry(
            max_missing_frames=(
                self.settings.worker_max_missing_frames
                if max_missing_frames is None
                else int(max_missing_frames)
            ),
            window=self.vote_config.window,
        )
        self.synthetic_ids = SyntheticTrackIds()
        self.stats = PipelineStats()
        self.frame_index = -1

    # -- introspection --------------------------------------------------
    @property
    def workers(self) -> list[WorkerState]:
        return self.registry.workers

    @property
    def active_violations(self) -> dict[int, frozenset[str]]:
        """Track ID -> currently committed violations."""
        return {w.track_id: frozenset(w.violations) for w in self.registry.workers}

    def worker(self, track_id: int) -> WorkerState | None:
        return self.registry.get(track_id)

    # -- main loop ------------------------------------------------------
    def step(self, frame: np.ndarray) -> FrameResult:
        """Advance the pipeline by one frame."""
        if self.detector is None:
            raise RuntimeError("FramePipeline has no detector")
        if frame is None:
            raise ValueError("frame must be a numpy array, got None")

        self.frame_index += 1
        self.stats.frames += 1

        detections = self._detect(frame)
        self.stats.detections += len(detections)

        person_detections = persons(detections)
        self.stats.persons += len(person_detections)

        report = associate(
            person_detections,
            detections,
            iou_min=self.iou_min,
            containment_min=self.containment_min,
        )
        self.stats.associations += len(report.associations)
        self.stats.bound_ppe += report.bound
        self.stats.unassigned_ppe += len(report.unassigned)

        vote_results: list[VoteResult] = []
        events: list[ViolationEvent] = []
        touched: list[WorkerState] = []

        for association in report.associations:
            identified = self._ensure_track_id(association)
            if identified.track_id is None:  # pragma: no cover - defensive
                self.stats.unseen_persons += 1
                continue

            worker = self.registry.touch(
                identified.track_id, identified.box, self.frame_index
            )
            worker.observe(identified, self.frame_index)
            touched.append(worker)

            required = self.policy.required_slots(identified.box)
            observed = frozenset(
                slug
                for slug in identified.violation_slugs
                if ppe_slot(slug) in required
            )

            result = self.voter.update(worker, observed)
            vote_results.append(result)
            self.stats.committed += len(result.committed)
            self.stats.cleared += len(result.cleared)

            if result.committed:
                events.append(self._build_event(worker, identified, result))

        self._publish(events)
        self.stats.workers_seen = len(self.registry)
        self.stats.workers_evicted += len(self.registry.evict_stale(self.frame_index))

        return FrameResult(
            frame_index=self.frame_index,
            detections=tuple(detections),
            association=report,
            votes=tuple(vote_results),
            events=tuple(events),
            workers=tuple(touched),
        )

    def run(self, frames: Iterable[np.ndarray]) -> list[FrameResult]:
        """Run ``step`` over an iterable of frames."""
        return [self.step(frame) for frame in frames]

    def events_from(self, results: Iterable[FrameResult]) -> list[ViolationEvent]:
        """Flatten every event produced by a sequence of frame results."""
        return [event for result in results for event in result.events]

    def reset(self) -> None:
        """Forget all workers and counters (call between independent videos)."""
        self.registry.clear()
        self.voter.reset()
        self.synthetic_ids.reset()
        self.stats = PipelineStats()
        self.frame_index = -1
        if self.detector is not None:
            self.detector.reset_tracker()

    # -- internals ------------------------------------------------------
    def _detect(self, frame: np.ndarray) -> list[Detection]:
        assert self.detector is not None  # guarded by step()
        if self.track:
            return self.detector.track(frame, iou=self.settings.iou)
        return self.detector.detect(frame, iou=self.settings.iou)

    def _ensure_track_id(self, association: PersonAssociation) -> PersonAssociation:
        """Give untracked persons a synthetic ID so keying is always total.

        In ``track=True`` mode ByteTrack supplies IDs and this is a no-op.  In
        ``track=False`` debug mode every person gets a fresh negative ID, which
        means no identity across frames and therefore no commits -- voting
        correctly refuses to fire without evidence of persistence.
        """
        if association.track_id is not None:
            return association
        synthetic = self.synthetic_ids.next()
        return relocate_track_id(association, synthetic)

    def _build_event(
        self,
        worker: WorkerState,
        association: PersonAssociation,
        result: VoteResult,
    ) -> ViolationEvent:
        confidence = association.violation_confidence
        if confidence <= 0.0:
            confidence = self._confidence_from_slots(worker, result.committed)
        zone = self.policy.zone_for(association.box)
        return ViolationEvent(
            track_id=worker.track_id,
            worker_box=association.box,
            violations=result.committed,
            frame_index=self.frame_index,
            timestamp=float(self.clock()),
            confidence=confidence,
            zone=zone,
            camera=self.camera,
        )

    @staticmethod
    def _confidence_from_slots(worker: WorkerState, slugs: Iterable[str]) -> float:
        """Fall back to the bound detections' confidence when the association is gone."""
        confidences: list[float] = []
        for slug in slugs:
            slot = ppe_slot(slug)
            if slot is None:
                continue
            detection = worker.slot(slot)
            if detection is not None:
                confidences.append(float(detection.confidence))
        if not confidences:
            return 0.0
        return float(sum(confidences) / len(confidences))

    def _publish(self, events: list[ViolationEvent]) -> None:
        if self.bus is not None and events:
            self.bus.publish_all(events)

    def as_dict(self) -> dict[str, Any]:
        """Pipeline snapshot: config, counters, worker states."""
        return {
            "frame_index": self.frame_index,
            "track": self.track,
            "vote_config": self.vote_config.as_dict(),
            "policy": type(self.policy).__name__,
            "stats": self.stats.as_dict(),
            "workers": self.registry.as_dict(),
        }


def build_pipeline(
    detector: Detector | None = None,
    *,
    settings: Settings | None = None,
    **kwargs: Any,
) -> FramePipeline:
    """Convenience factory used by the CLI."""
    return FramePipeline(detector, settings=settings, **kwargs)


__all__ = [
    "FramePipeline",
    "FrameResult",
    "GlobalPolicy",
    "PERSON_CLASS",
    "PipelineStats",
    "Policy",
    "build_pipeline",
    "with_track_id",
]


# Guard rails: these constants exist so a config drift is caught by tests
# rather than by a silently mis-tuned deployment.
DEFAULTS_IN_USE = {
    "vote_window": VOTE_WINDOW,
    "vote_threshold": DEFAULT_VOTE_CONFIG.threshold,
    "clear_hysteresis": DEFAULT_VOTE_CONFIG.clear_hysteresis,
    "association_iou_min": ASSOCIATION_IOU_MIN,
    "association_containment_min": ASSOCIATION_CONTAINMENT_MIN,
    "worker_max_missing_frames": WORKER_MAX_MISSING_FRAMES,
}
