"""Violation events and the in-process event bus.

A :class:`ViolationEvent` is the publishable unit: a track-bound, timestamped,
policy-filtered statement that a specific worker was non-compliant in a
specific frame.  Everything downstream -- the hash-chained audit log, the
snapshot archive, the alert router, the WebSocket feed -- consumes these and
nothing else, which is why they are frozen and content-addressed.

The bus here is deliberately in-process (thread-safe ``queue.Queue``): Phase 2
must run and be tested with no Redis.  Phase 4 adds a Redis Streams transport
behind the same ``publish`` / ``subscribe`` surface.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from .tracking import Box


def _content_hash(payload: dict[str, Any]) -> str:
    """Stable sha256 over the event's semantic content."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ViolationEvent:
    """A committed, track-bound compliance violation.

    ``event_id`` is derived from the event's content, so the same violation
    observed twice produces the same ID: downstream consumers get idempotent
    deduplication for free, and Phase 4's hash chain can detect a rewritten
    event by recomputing it.
    """

    track_id: int
    worker_box: Box
    violations: frozenset[str]
    frame_index: int
    timestamp: float
    confidence: float
    zone: str | None = None
    camera: str | None = None
    severity: str = "violation"
    event_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "violations", frozenset(self.violations))
        if not self.event_id:
            object.__setattr__(self, "event_id", self.compute_id())

    def content(self) -> dict[str, Any]:
        """The semantic content that defines event identity."""
        return {
            "track_id": self.track_id,
            "worker_box": self.worker_box.as_list(),
            "violations": sorted(self.violations),
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "confidence": round(float(self.confidence), 6),
            "zone": self.zone,
            "camera": self.camera,
            "severity": self.severity,
        }

    def compute_id(self) -> str:
        """Recompute the content hash, ignoring any stored ``event_id``."""
        return _content_hash(self.content())

    def is_intact(self) -> bool:
        """Whether ``event_id`` still matches the content (tamper check)."""
        return self.event_id == self.compute_id()

    @property
    def classes(self) -> tuple[str, ...]:
        """Violation classes, sorted for deterministic output."""
        return tuple(sorted(self.violations))

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view, with ``violations`` as a sorted list."""
        data = self.content()
        data["event_id"] = self.event_id
        data["violations"] = sorted(self.violations)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ViolationEvent:
        """Rebuild an event from :meth:`to_dict` output.

        A stored ``event_id`` is preserved verbatim (so a corrupted ID survives
        a round trip and stays detectable via :meth:`is_intact`).
        """
        raw_box = data.get("worker_box") or (0.0, 0.0, 0.0, 0.0)
        return cls(
            track_id=int(data["track_id"]),
            worker_box=Box.from_xyxy(raw_box),
            violations=frozenset(data.get("violations") or ()),
            frame_index=int(data["frame_index"]),
            timestamp=float(data["timestamp"]),
            confidence=float(data["confidence"]),
            zone=data.get("zone"),
            camera=data.get("camera"),
            severity=str(data.get("severity", "violation")),
            event_id=str(data.get("event_id", "")),
        )

    @classmethod
    def from_json(cls, raw: str) -> ViolationEvent:
        """Rebuild an event from :meth:`to_json` output."""
        return cls.from_dict(json.loads(raw))


@dataclass
class BusStats:
    """Counters exposed for tests, logging and Prometheus (Phase 7)."""

    published: int = 0
    delivered: int = 0
    dropped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "published": self.published,
            "delivered": self.delivered,
            "dropped": self.dropped,
        }


class EventBus:
    """Thread-safe in-process fan-out of :class:`ViolationEvent`.

    Each subscriber gets its own queue, so one slow consumer can never stall
    another.  ``maxsize`` applies per subscriber: when a queue is full the
    event is dropped and counted rather than blocking the pipeline -- dropping
    a dashboard frame is survivable, stalling inference is not.
    """

    def __init__(self, *, maxsize: int = 0, recent: int = 500) -> None:
        self.maxsize = int(maxsize)
        self._subscribers: list[queue.Queue[ViolationEvent]] = []
        self._recent: deque[ViolationEvent] = deque(maxlen=max(0, int(recent)))
        self._lock = threading.Lock()
        self.stats = BusStats()

    # -- subscription ---------------------------------------------------
    def subscribe(self, maxsize: int | None = None) -> queue.Queue[ViolationEvent]:
        """Register a new subscriber and return its queue."""
        subscriber: queue.Queue[ViolationEvent] = queue.Queue(
            maxsize=self.maxsize if maxsize is None else int(maxsize)
        )
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[ViolationEvent]) -> bool:
        """Remove a subscriber; ``True`` if it was registered."""
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
                return True
        return False

    @property
    def subscribers(self) -> list[queue.Queue[ViolationEvent]]:
        with self._lock:
            return list(self._subscribers)

    @property
    def subscriber_count(self) -> int:
        return len(self.subscribers)

    @property
    def published_count(self) -> int:
        return self.stats.published

    # -- publication ----------------------------------------------------
    def publish(self, event: ViolationEvent) -> int:
        """Deliver ``event`` to every subscriber; return the delivery count."""
        self.stats.published += 1
        self._recent.append(event)
        delivered = 0
        for subscriber in self.subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                self.stats.dropped += 1
            else:
                delivered += 1
        self.stats.delivered += delivered
        return delivered

    def publish_all(self, events: Any) -> int:
        """Publish an iterable of events; return total deliveries."""
        return sum(self.publish(event) for event in events)

    def drain(self, subscriber: queue.Queue[ViolationEvent] | None = None) -> list[ViolationEvent]:
        """Non-blocking drain of one subscriber, or of the first subscriber."""
        target = subscriber
        if target is None:
            subscribers = self.subscribers
            if not subscribers:
                return []
            target = subscribers[0]
        collected: list[ViolationEvent] = []
        while True:
            try:
                collected.append(target.get_nowait())
            except queue.Empty:
                return collected

    def recent(self, limit: int | None = None) -> list[ViolationEvent]:
        """Most recent events, newest first."""
        items = list(self._recent)
        items.reverse()
        if limit is not None:
            return items[: max(0, int(limit))]
        return items

    def close(self) -> None:
        """Remove every subscriber."""
        with self._lock:
            self._subscribers.clear()

    def as_dict(self) -> dict[str, Any]:
        return {
            "subscribers": self.subscriber_count,
            "recent": len(self._recent),
            **self.stats.as_dict(),
        }
