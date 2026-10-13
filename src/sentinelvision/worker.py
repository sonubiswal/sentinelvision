"""Per-worker state: what we believe about one tracked worker.

A :class:`WorkerState` is the memory between frames.  It holds the current
person box, the detection bound to each PPE slot, the per-frame violation
history the voter reads, and the set of violations currently *committed*.

:class:`WorkerRegistry` owns the lifetime of those objects: creation on first
sight, update on every sighting, eviction after a configurable number of missed
frames.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .association import (
    PPE_SLOTS,
    PersonAssociation,
    SlotObservation,
    SlotState,
)
from .classes import normalize_class_name
from .config import VOTE_WINDOW, WORKER_MAX_MISSING_FRAMES
from .detector import Detection
from .tracking import Box

#: Slot slug -> attribute name on :class:`WorkerState`.
SLOT_ATTRIBUTES: dict[str, str] = {
    "hardhat": "hardhat",
    "safety-vest": "safety_vest",
    "mask": "mask",
}


@dataclass
class WorkerState:
    """Everything SentinelVision knows about one tracked worker."""

    track_id: int
    person_box: Box
    hardhat: Detection | None = None
    safety_vest: Detection | None = None
    mask: Detection | None = None
    #: Violations currently committed (i.e. past the N-of-M vote).
    violations: set[str] = field(default_factory=set)
    #: Per-frame violation sets, newest last.  Bounded by ``window``.
    history: deque[set[str]] = field(default=None, repr=False)  # type: ignore[assignment]
    #: Consecutive frames each committed violation has been absent.
    clean_streak: dict[str, int] = field(default_factory=dict)
    #: Latest slot states, for dashboards and debugging.
    slot_states: dict[str, SlotState] = field(default_factory=dict)
    window: int = VOTE_WINDOW
    frames_seen: int = 0
    missing_frames: int = 0
    first_seen_frame: int = -1
    last_seen_frame: int = -1

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = deque(maxlen=self.window)
        if self.slot_states == {}:
            self.slot_states = {slot: SlotState.UNKNOWN for slot in PPE_SLOTS}

    # -- slot access ----------------------------------------------------
    def slot(self, name: str) -> Detection | None:
        """The detection bound to PPE ``name``, or ``None``.

        Raises ``ValueError`` for a name that is not a configured slot, rather
        than silently returning ``None`` for a typo.
        """
        attribute = SLOT_ATTRIBUTES.get(normalize_class_name(name))
        if attribute is None:
            raise ValueError(
                f"unknown PPE slot: {name!r}; known slots: {sorted(SLOT_ATTRIBUTES)}"
            )
        return getattr(self, attribute)

    def set_slot(self, name: str, detection: Detection | None) -> None:
        """Bind ``detection`` (or ``None``) to PPE slot ``name``."""
        attribute = SLOT_ATTRIBUTES.get(normalize_class_name(name))
        if attribute is None:
            raise ValueError(
                f"unknown PPE slot: {name!r}; known slots: {sorted(SLOT_ATTRIBUTES)}"
            )
        setattr(self, attribute, detection)

    def state(self, name: str) -> SlotState:
        """Latest state of PPE slot ``name``."""
        return self.slot_states.get(normalize_class_name(name), SlotState.UNKNOWN)

    # -- frame bookkeeping ---------------------------------------------
    def observe(self, association: PersonAssociation, frame_index: int) -> None:
        """Fold one frame's association into this worker's state."""
        self.person_box = association.box
        self.frames_seen += 1
        self.missing_frames = 0
        if self.first_seen_frame < 0:
            self.first_seen_frame = frame_index
        self.last_seen_frame = frame_index

        for slot, observation in association.slots.items():
            self._apply_observation(slot, observation)

    def _apply_observation(self, slot: str, observation: SlotObservation) -> None:
        slug = normalize_class_name(slot)
        self.slot_states[slug] = observation.state
        if slug in SLOT_ATTRIBUTES:
            attribute = SLOT_ATTRIBUTES[slug]
            if observation.state is SlotState.UNKNOWN:
                # Occluded or not visible: forget the evidence rather than
                # claiming "no hardhat".
                setattr(self, attribute, None)
            else:
                setattr(self, attribute, observation.detection)

    def record_frame(self, frame_violations: set[str]) -> None:
        """Append this frame's (policy-filtered) violation set to the history."""
        self.history.append(set(frame_violations))

    @property
    def frame_violations(self) -> set[str]:
        """Violation classes observed in the *latest* frame, from slot states."""
        return {
            f"no-{slot}"
            for slot, state in self.slot_states.items()
            if state is SlotState.VIOLATION
        }

    @property
    def observations(self) -> int:
        """How many frames have contributed to the history."""
        return len(self.history)

    @property
    def unknown_slots(self) -> frozenset[str]:
        return frozenset(
            slot for slot, state in self.slot_states.items() if state is SlotState.UNKNOWN
        )

    @property
    def present_slots(self) -> frozenset[str]:
        return frozenset(
            slot for slot, state in self.slot_states.items() if state is SlotState.PRESENT
        )

    @property
    def violation_slots(self) -> frozenset[str]:
        return frozenset(
            slot for slot, state in self.slot_states.items() if state is SlotState.VIOLATION
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly snapshot for the dashboard and for tests."""
        return {
            "track_id": self.track_id,
            "box": self.person_box.as_list(),
            "slots": {
                slot: {
                    "state": self.slot_states.get(slot, SlotState.UNKNOWN).value,
                    "class": det.class_name if (det := self.slot(slot)) is not None else None,
                    "confidence": float(det.confidence) if det is not None else 0.0,
                }
                for slot in PPE_SLOTS
            },
            "violations": sorted(self.violations),
            "frames_seen": self.frames_seen,
            "missing_frames": self.missing_frames,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "observations": self.observations,
        }


class WorkerRegistry:
    """Dict-like store of :class:`WorkerState` keyed by track ID."""

    def __init__(
        self,
        *,
        max_missing_frames: int = WORKER_MAX_MISSING_FRAMES,
        window: int = VOTE_WINDOW,
    ) -> None:
        self.max_missing_frames = int(max_missing_frames)
        self.window = int(window)
        self._workers: dict[int, WorkerState] = {}

    # -- container protocol --------------------------------------------
    def __len__(self) -> int:
        return len(self._workers)

    def __contains__(self, track_id: object) -> bool:
        return track_id in self._workers

    def __iter__(self) -> Iterator[WorkerState]:
        # Sorted by track ID so iteration order never depends on dict insertion
        # order (i.e. on the order workers happened to appear in the stream).
        return iter([self._workers[k] for k in sorted(self._workers)])

    def __getitem__(self, track_id: int) -> WorkerState:
        return self._workers[track_id]

    def get(self, track_id: int) -> WorkerState | None:
        return self._workers.get(track_id)

    @property
    def track_ids(self) -> list[int]:
        return sorted(self._workers)

    @property
    def workers(self) -> list[WorkerState]:
        """Workers ordered by track ID (deterministic across runs)."""
        return [self._workers[k] for k in sorted(self._workers)]

    # -- lifecycle ------------------------------------------------------
    def touch(self, track_id: int, person_box: Box, frame_index: int) -> WorkerState:
        """Return the worker for ``track_id``, creating it on first sight.

        Both ``first_seen_frame`` and ``last_seen_frame`` are stamped here.
        ``last_seen_frame`` in particular must be set even when ``observe`` is
        never called, because :meth:`evict_stale` uses it to tell a live worker
        from an absent one -- a stale stamp evicts workers that are still on
        screen.
        """
        worker = self._workers.get(track_id)
        if worker is None:
            worker = WorkerState(
                track_id=track_id,
                person_box=person_box,
                window=self.window,
                first_seen_frame=frame_index,
                last_seen_frame=frame_index,
            )
            self._workers[track_id] = worker
        else:
            worker.person_box = person_box
            worker.missing_frames = 0
            worker.last_seen_frame = frame_index
            if worker.first_seen_frame < 0:
                worker.first_seen_frame = frame_index
        return worker

    def drop(self, track_id: int) -> None:
        """Forget a worker immediately."""
        self._workers.pop(track_id, None)

    def clear(self) -> None:
        """Forget every worker."""
        self._workers.clear()

    def evict_stale(self, frame_index: int) -> list[int]:
        """Age out workers not seen at ``frame_index``; return removed IDs."""
        removed: list[int] = []
        for track_id, worker in self._workers.items():
            if worker.last_seen_frame == frame_index:
                continue
            worker.missing_frames += 1
            if worker.missing_frames > self.max_missing_frames:
                removed.append(track_id)
        for track_id in removed:
            del self._workers[track_id]
        return sorted(removed)

    def as_dict(self) -> dict[int, dict[str, Any]]:
        """Every worker as a JSON-friendly snapshot."""
        return {worker.track_id: worker.as_dict() for worker in self.workers}
