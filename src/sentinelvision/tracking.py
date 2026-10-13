"""Geometry primitives and track-ID plumbing.

Everything here is pure geometry: no model, no I/O, no state.  The association
layer (``association.py``) and the pipeline build on these primitives, and the
test suite exercises them directly with hand-computed boxes.

Coordinate convention: ``(x1, y1)`` is the top-left corner, ``(x2, y2)`` the
bottom-right, both in pixels, both inclusive at the corner.  ``x2 > x1`` and
``y2 > y1`` for any real box.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from .classes import normalize_class_name
from .detector import Detection

#: The single class name that denotes a worker.
PERSON_CLASS = "person"

EPSILON = 1e-9


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectangle in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xyxy(cls, xyxy: Iterable[float]) -> Box:
        """Build a Box from any 4-element iterable."""
        values = tuple(float(v) for v in xyxy)
        if len(values) != 4:
            raise ValueError(f"expected 4 coordinates, got {len(values)}")
        return cls(values[0], values[1], values[2], values[3])

    @classmethod
    def from_detection(cls, detection: Detection) -> Box:
        """Build a Box from a :class:`~sentinelvision.detector.Detection`."""
        return cls.from_xyxy(detection.xyxy)

    @classmethod
    def from_center(cls, center: tuple[float, float], width: float, height: float) -> Box:
        """Build a Box from a centre point and a size (handy in tests)."""
        cx, cy = float(center[0]), float(center[1])
        half_w, half_h = float(width) / 2.0, float(height) / 2.0
        return cls(cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)

    @property
    def diagonal(self) -> float:
        """Length of the box diagonal -- used to normalise distances."""
        return float((self.width**2 + self.height**2) ** 0.5)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def intersection_area(self, other: Box) -> float:
        """Area of overlap with ``other``; ``0.0`` when disjoint."""
        left = max(self.x1, other.x1)
        top = max(self.y1, other.y1)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)
        if right <= left or bottom <= top:
            return 0.0
        return float((right - left) * (bottom - top))

    def union_area(self, other: Box) -> float:
        return self.area + other.area - self.intersection_area(other)

    def iou(self, other: Box) -> float:
        """Intersection-over-union.  ``0.0`` when either box has no area."""
        union = self.union_area(other)
        if union <= EPSILON:
            return 0.0
        return float(self.intersection_area(other) / union)

    def contains_point(self, x: float, y: float, *, inclusive: bool = True) -> bool:
        """Whether ``(x, y)`` lies inside the box.

        ``inclusive`` treats the edges as inside, which matters for a PPE box
        whose centre lands exactly on a person-box edge.
        """
        if inclusive:
            return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
        return self.x1 < x < self.x2 and self.y1 < y < self.y2


def box_area(box: Box) -> float:
    """Area of ``box``."""
    return box.area


def box_center(box: Box) -> tuple[float, float]:
    """Centre point of ``box``."""
    return box.center


def box_iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes."""
    return a.iou(b)


def containment(inner: Box, outer: Box) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``.

    ``1.0`` means ``inner`` is fully contained.  Unlike IoU this is not
    penalised by ``outer`` being much larger, which is exactly the case for a
    small hardhat box inside a person box.
    """
    if inner.area <= EPSILON:
        return 0.0
    return float(inner.intersection_area(outer) / inner.area)


def center_inside(inner: Box, outer: Box) -> bool:
    """Whether ``inner``'s centre point falls inside ``outer``."""
    cx, cy = inner.center
    return outer.contains_point(cx, cy)


def center_distance(a: Box, b: Box) -> float:
    """Euclidean distance between two box centres, in pixels."""
    ax, ay = a.center
    bx, by = b.center
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def normalized_center_distance(reference: Box, a: Box, b: Box) -> float:
    """Centre distance normalised by ``reference``'s diagonal.

    Makes the tie-break scale-invariant: a 40 px gap is decisive for a small
    box and negligible for a huge one.
    """
    denominator = max(reference.diagonal, EPSILON)
    return center_distance(a, b) / denominator


# --- class helpers ----------------------------------------------------------
def is_person(detection: Detection) -> bool:
    """Whether the detection is a worker."""
    return detection.class_name == PERSON_CLASS


def persons(detections: Iterable[Detection]) -> list[Detection]:
    """Only the person detections, in input order."""
    return [d for d in detections if is_person(d)]


def non_persons(detections: Iterable[Detection]) -> list[Detection]:
    """Everything that is not a person (PPE + context classes)."""
    return [d for d in detections if not is_person(d)]


def group_by_class(detections: Iterable[Detection]) -> dict[str, list[Detection]]:
    """Group detections by normalized class slug, preserving input order."""
    grouped: dict[str, list[Detection]] = {}
    for det in detections:
        grouped.setdefault(normalize_class_name(det.class_name), []).append(det)
    return grouped


def with_track_id(detection: Detection, track_id: int | None) -> Detection:
    """Return a copy of ``detection`` carrying ``track_id``.

    :class:`Detection` is frozen, so identity enrichment produces a new object
    and never mutates the input.
    """
    return replace(detection, track_id=track_id)


def with_track_ids(detections: Iterable[Detection], track_ids: Iterable[int | None]) -> list[Detection]:
    """Apply ``with_track_id`` pairwise over two iterables."""
    return [
        with_track_id(det, tid) for det, tid in zip(detections, track_ids, strict=False)
    ]


class SyntheticTrackIds:
    """Allocates negative track IDs for detections that arrive untracked.

    Negative IDs guarantee no collision with real ByteTrack IDs (which start at
    1).  Used only in ``track=False`` debug mode: without a real tracker there
    is no cross-frame identity, so each frame's untracked person gets a fresh
    ID and temporal voting correctly declines to commit anything.
    """

    def __init__(self, start: int = -1) -> None:
        self._next = int(start)

    def next(self) -> int:
        value = self._next
        self._next -= 1
        return value

    def reset(self) -> None:
        self._next = -1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SyntheticTrackIds(next={self._next})"


def max_box_area(boxes: Iterable[Box]) -> float:
    """Largest area among ``boxes`` (``0.0`` when empty)."""
    return max((box.area for box in boxes), default=0.0)


def as_dict(box: Box) -> dict[str, Any]:  # pragma: no cover - serialization helper
    """JSON-friendly view of a box."""
    return {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2}
