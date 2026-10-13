"""ViolationEvent identity/tamper detection and the in-process event bus."""

from __future__ import annotations

import json
import queue
from dataclasses import FrozenInstanceError, replace

import pytest

from sentinelvision.events import BusStats, EventBus, ViolationEvent
from sentinelvision.tracking import Box

BOX = Box(0.0, 0.0, 100.0, 200.0)


def event(**overrides) -> ViolationEvent:
    payload = {
        "track_id": 1,
        "worker_box": BOX,
        "violations": frozenset({"no-hardhat"}),
        "frame_index": 10,
        "timestamp": 1700000000.5,
        "confidence": 0.87,
    }
    payload.update(overrides)
    return ViolationEvent(**payload)


# --- identity --------------------------------------------------------------
def test_event_id_is_content_addressed() -> None:
    assert event().event_id == event().event_id
    assert event().event_id == event().compute_id()
    assert len(event().event_id) == 64  # sha256 hex


def test_event_content_shapes_the_id() -> None:
    base = event()
    assert base.event_id != event(track_id=2).event_id
    assert base.event_id != event(confidence=0.88).event_id
    assert base.event_id != event(frame_index=11).event_id
    assert base.event_id != event(violations=frozenset({"no-mask"})).event_id
    assert base.event_id != event(zone="bay-1").event_id


def test_event_id_ignores_violation_ordering() -> None:
    a = event(violations=frozenset({"no-hardhat", "no-mask"}))
    b = event(violations=frozenset({"no-mask", "no-hardhat"}))
    assert a.event_id == b.event_id


def test_violations_are_coerced_to_a_frozenset() -> None:
    evt = event(violations=["no-hardhat", "no-mask"])
    assert isinstance(evt.violations, frozenset)
    assert evt.classes == ("no-hardhat", "no-mask")


def test_event_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        event().track_id = 99  # type: ignore[misc]


# --- tamper detection ------------------------------------------------------
def test_pristine_event_reports_intact() -> None:
    assert event().is_intact() is True


def test_rewriting_content_while_keeping_the_id_is_detected() -> None:
    """The property Phase 4's hash chain is built on."""
    original = event()
    forged = replace(original, confidence=0.99)
    assert forged.event_id == original.event_id  # id carried over
    assert forged.is_intact() is False


def test_tampered_serialized_event_is_detected() -> None:
    data = json.loads(event().to_json())
    data["violations"] = ["no-mask"]
    rebuilt = ViolationEvent.from_dict(data)
    assert rebuilt.violations == frozenset({"no-mask"})
    assert rebuilt.is_intact() is False


# --- serialization ---------------------------------------------------------
def test_round_trip_through_dict_and_json() -> None:
    original = event(zone="welding-bay", camera="cam-01")
    from_dict = ViolationEvent.from_dict(original.to_dict())
    from_json = ViolationEvent.from_json(original.to_json())

    for rebuilt in (from_dict, from_json):
        assert rebuilt.event_id == original.event_id
        assert rebuilt.violations == original.violations
        assert rebuilt.worker_box == original.worker_box
        assert rebuilt.track_id == original.track_id
        assert rebuilt.zone == original.zone
        assert rebuilt.camera == original.camera
        assert rebuilt.severity == original.severity
        assert rebuilt.is_intact() is True


def test_to_dict_is_json_friendly() -> None:
    data = event().to_dict()
    assert data["violations"] == ["no-hardhat"]  # sorted list, not a set
    assert data["worker_box"] == [0.0, 0.0, 100.0, 200.0]
    assert json.loads(json.dumps(data))["event_id"] == event().event_id


def test_defaults_for_optional_fields() -> None:
    evt = event()
    assert evt.zone is None
    assert evt.camera is None
    assert evt.severity == "violation"


# --- bus -------------------------------------------------------------------
def test_publish_reaches_every_subscriber() -> None:
    bus = EventBus()
    first, second = bus.subscribe(), bus.subscribe()
    evt = event()
    assert bus.publish(evt) == 2
    assert bus.drain(first) == [evt]
    assert bus.drain(second) == [evt]
    assert bus.published_count == 1
    assert bus.stats.delivered == 2


def test_publish_with_no_subscribers_counts_the_publication() -> None:
    bus = EventBus()
    assert bus.publish(event()) == 0
    assert bus.stats.published == 1
    assert bus.stats.delivered == 0


def test_publish_all_returns_total_deliveries() -> None:
    bus = EventBus()
    bus.subscribe()
    assert bus.publish_all([event(frame_index=1), event(frame_index=2)]) == 2
    assert len(bus.recent()) == 2


def test_drain_on_an_empty_bus_is_empty() -> None:
    assert EventBus().drain() == []


def test_drain_defaults_to_the_first_subscriber() -> None:
    bus = EventBus()
    bus.subscribe()
    bus.publish(event())
    assert len(bus.drain()) == 1
    assert bus.drain() == []


def test_a_slow_subscriber_drops_rather_than_stalling_the_pipeline() -> None:
    bus = EventBus()
    slow = bus.subscribe(maxsize=1)
    fast = bus.subscribe()

    bus.publish(event(frame_index=1))
    bus.publish(event(frame_index=2))

    assert bus.stats.dropped == 1
    assert len(bus.drain(slow)) == 1  # only the first survived
    assert len(bus.drain(fast)) == 2  # unbounded subscriber got both


def test_subscribers_are_isolated_from_each_other() -> None:
    bus = EventBus()
    first, second = bus.subscribe(maxsize=1), bus.subscribe(maxsize=1)
    bus.publish(event(frame_index=1))
    bus.drain(first)
    # Draining one subscriber must not disturb the other's queue.
    assert len(bus.drain(second)) == 1


def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    subscriber = bus.subscribe()
    assert bus.subscriber_count == 1
    assert bus.unsubscribe(subscriber) is True
    assert bus.unsubscribe(subscriber) is False
    assert bus.publish(event()) == 0
    assert bus.subscriber_count == 0


def test_close_removes_every_subscriber() -> None:
    bus = EventBus()
    bus.subscribe()
    bus.subscribe()
    bus.close()
    assert bus.subscriber_count == 0


def test_recent_is_newest_first_and_bounded() -> None:
    bus = EventBus(recent=2)
    bus.publish(event(frame_index=1))
    bus.publish(event(frame_index=2))
    bus.publish(event(frame_index=3))
    recent = bus.recent()
    assert [evt.frame_index for evt in recent] == [3, 2]
    assert [evt.frame_index for evt in bus.recent(1)] == [3]


def test_recent_ignores_dropped_subscriber_queues() -> None:
    bus = EventBus()
    bus.subscribe(maxsize=1)
    bus.publish(event(frame_index=1))
    bus.publish(event(frame_index=2))
    assert len(bus.recent()) == 2  # history is independent of delivery


def test_subscribers_property_returns_a_copy() -> None:
    bus = EventBus()
    subscriber = bus.subscribe()
    snapshot = bus.subscribers
    snapshot.clear()
    assert bus.subscriber_count == 1
    assert bus.unsubscribe(subscriber) is True


def test_bus_stats_dict() -> None:
    bus = EventBus()
    bus.subscribe()
    bus.publish(event())
    assert bus.as_dict()["published"] == 1
    assert bus.as_dict()["subscribers"] == 1
    assert BusStats().as_dict() == {"published": 0, "delivered": 0, "dropped": 0}


def test_subscriber_queue_type() -> None:
    assert isinstance(EventBus().subscribe(), queue.Queue)
