"""Person <-> PPE association.

The single most important idea in this file: a PPE slot has **three** states,
not two.

===========  ===============================  ==============================
State        Meaning                          Counts as a violation?
===========  ===============================  ==============================
``PRESENT``  A positive PPE detection bound    No -- compliant evidence
``VIOLATION``A ``NO-*`` detection bound        Yes
``UNKNOWN``  Nothing bound this frame         **No** -- occlusion, blur, motion
===========  ===============================  ==============================

Collapsing ``VIOLATION`` and ``UNKNOWN`` into one bucket is the single largest
false-positive source in detection-only PPE repos: a worker walking behind a
pallet disappears from the model's view and is instantly "not wearing a
hardhat".  SentinelVision keeps the third state, and the voter downstream only
ever counts ``VIOLATION``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from .classes import (
    POSITIVE_PPE_CLASSES,
    VIOLATION_PREFIX,
    normalize_class_name,
)
from .config import ASSOCIATION_CONTAINMENT_MIN, ASSOCIATION_IOU_MIN
from .detector import Detection
from .tracking import (
    Box,
    center_inside,
    containment,
    with_track_id,
)


class SlotState(str, Enum):
    """The three states a PPE slot on a worker can be in."""

    PRESENT = "present"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


#: Every PPE slot this platform tracks, in taxonomy order.
PPE_SLOTS: tuple[str, ...] = tuple(POSITIVE_PPE_CLASSES)


def ppe_slot(class_name: str) -> str | None:
    """Map a class name to the PPE slot it belongs to.

    ``"Hardhat"`` -> ``"hardhat"``; ``"NO-Safety Vest"`` -> ``"safety-vest"``;
    ``"machinery"`` -> ``None`` (context, not PPE).
    """
    slug = normalize_class_name(class_name)
    base = slug[len(VIOLATION_PREFIX) :] if slug.startswith(VIOLATION_PREFIX) else slug
    return base if base in PPE_SLOTS else None


def violation_slug_for_slot(slot: str) -> str:
    """``"hardhat"`` -> ``"no-hardhat"`` (the violation class for that slot)."""
    return f"{VIOLATION_PREFIX}{normalize_class_name(slot)}"


def is_ppe(detection: Detection) -> bool:
    """Whether the detection belongs to a tracked PPE slot."""
    return ppe_slot(detection.class_name) is not None


@dataclass(frozen=True)
class SlotObservation:
    """What was seen for one PPE slot on one worker, in one frame."""

    slot: str
    state: SlotState
    detection: Detection | None = None
    person_track_id: int | None = None

    @property
    def is_violation(self) -> bool:
        return self.state is SlotState.VIOLATION

    @property
    def is_present(self) -> bool:
        return self.state is SlotState.PRESENT

    @property
    def is_unknown(self) -> bool:
        return self.state is SlotState.UNKNOWN

    @property
    def confidence(self) -> float:
        """Confidence of the bound detection; ``0.0`` when nothing was bound."""
        return float(self.detection.confidence) if self.detection is not None else 0.0

    @property
    def violation_slug(self) -> str:
        """The violation class this slot would produce."""
        return violation_slug_for_slot(self.slot)

    def as_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "state": self.state.value,
            "confidence": self.confidence,
            "class": self.detection.class_name if self.detection is not None else None,
        }


@dataclass(frozen=True)
class PersonAssociation:
    """One person and the PPE slots bound to them for this frame."""

    person: Detection
    track_id: int | None
    slots: dict[str, SlotObservation]

    @property
    def box(self) -> Box:
        return Box.from_detection(self.person)

    @property
    def confidence(self) -> float:
        return float(self.person.confidence)

    def state(self, slot: str) -> SlotState:
        """State of ``slot``; an absent slot is ``UNKNOWN``, never a violation."""
        observation = self.slots.get(normalize_class_name(slot))
        return observation.state if observation is not None else SlotState.UNKNOWN

    def detection(self, slot: str) -> Detection | None:
        """The detection bound to ``slot``, if any."""
        observation = self.slots.get(normalize_class_name(slot))
        return observation.detection if observation is not None else None

    @property
    def violation_slugs(self) -> frozenset[str]:
        """Violation classes to feed into temporal voting."""
        return frozenset(
            obs.violation_slug for obs in self.slots.values() if obs.state is SlotState.VIOLATION
        )

    @property
    def present_slots(self) -> frozenset[str]:
        return frozenset(s for s, o in self.slots.items() if o.state is SlotState.PRESENT)

    @property
    def unknown_slots(self) -> frozenset[str]:
        return frozenset(s for s, o in self.slots.items() if o.state is SlotState.UNKNOWN)

    @property
    def violation_confidence(self) -> float:
        """Mean confidence across bound violation detections (``0.0`` if none)."""
        hits = [o.confidence for o in self.slots.values() if o.state is SlotState.VIOLATION]
        return float(sum(hits) / len(hits)) if hits else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "box": self.box.as_list(),
            "person_confidence": self.confidence,
            "slots": {name: obs.as_dict() for name, obs in self.slots.items()},
            "violations": sorted(self.violation_slugs),
        }


@dataclass(frozen=True)
class AssociationReport:
    """Result of binding every PPE detection in a frame to the persons."""

    associations: tuple[PersonAssociation, ...]
    unassigned: tuple[Detection, ...]
    bound: int

    def __len__(self) -> int:
        return len(self.associations)

    def __iter__(self):
        return iter(self.associations)

    @property
    def violations(self) -> frozenset[str]:
        """Union of every violation class observed this frame."""
        found: set[str] = set()
        for association in self.associations:
            found |= association.violation_slugs
        return frozenset(found)

    def by_track_id(self) -> dict[int, PersonAssociation]:
        """Map track ID -> association, skipping untracked persons."""
        return {a.track_id: a for a in self.associations if a.track_id is not None}


def bind_ppe_to_person(
    ppe: Detection,
    persons_seq: Sequence[Detection],
    *,
    iou_min: float = ASSOCIATION_IOU_MIN,
    containment_min: float = ASSOCIATION_CONTAINMENT_MIN,
) -> Detection | None:
    """Find the person a PPE detection belongs to, or ``None``.

    A person is a candidate when any of these hold:

    * the PPE centre point lies inside the person box;
    * ``IoU(person, ppe) >= iou_min``;
    * at least ``containment_min`` of the PPE box area lies inside the person.

    With several candidates the *nearest* person by centre distance wins, and
    ties break deterministically on track ID then input order, so results never
    depend on set iteration order.
    """
    ppe_box = Box.from_detection(ppe)
    candidates: list[tuple[float, int, int, Detection]] = []

    for index, person in enumerate(persons_seq):
        person_box = Box.from_detection(person)
        binds = (
            center_inside(ppe_box, person_box)
            or person_box.iou(ppe_box) >= iou_min
            or containment(ppe_box, person_box) >= containment_min
        )
        if not binds:
            continue
        distance = _normalized_distance(ppe_box, person_box)
        track_key = person.track_id if person.track_id is not None else -1
        candidates.append((distance, track_key, index, person))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def _normalized_distance(a: Box, b: Box) -> float:
    """Centre distance divided by the mean diagonal -- scale invariant."""
    ax, ay = a.center
    bx, by = b.center
    raw = float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
    scale = max((a.diagonal + b.diagonal) / 2.0, 1e-9)
    return raw / scale


def _best_for_slot(detections: Sequence[Detection]) -> Detection | None:
    """Pick the winning detection when several bind to the same slot.

    Highest confidence wins.  On an exact tie the *positive* (compliant)
    detection wins, because a false violation costs more than a missed one:
    trust in the system is the scarce resource.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: (d.confidence, not d.is_violation))


def associate(
    persons_seq: Sequence[Detection],
    detections: Iterable[Detection],
    *,
    iou_min: float = ASSOCIATION_IOU_MIN,
    containment_min: float = ASSOCIATION_CONTAINMENT_MIN,
) -> AssociationReport:
    """Bind PPE detections to persons and classify every PPE slot.

    Every configured slot is emitted for every person, so a worker with no
    visible hardhat yields ``hardhat=UNKNOWN`` rather than a missing key.
    """
    ppe_detections = [d for d in detections if is_ppe(d)]

    # Keyed by object identity: persons_seq elements are never copied, so the
    # same objects reappear in the enumerate() loop below.
    per_person: dict[int, list[Detection]] = {id(person): [] for person in persons_seq}
    unassigned: list[Detection] = []

    for ppe in ppe_detections:
        owner = bind_ppe_to_person(
            ppe, persons_seq, iou_min=iou_min, containment_min=containment_min
        )
        if owner is None:
            unassigned.append(ppe)
            continue
        per_person[id(owner)].append(ppe)

    associations: list[PersonAssociation] = []
    bound = 0
    for person in persons_seq:
        bound_here = per_person.get(id(person), [])
        bound += len(bound_here)
        slots: dict[str, SlotObservation] = {}
        for slot in PPE_SLOTS:
            candidates = [d for d in bound_here if ppe_slot(d.class_name) == slot]
            winner = _best_for_slot(candidates)
            if winner is None:
                state = SlotState.UNKNOWN
            elif winner.is_violation:
                state = SlotState.VIOLATION
            else:
                state = SlotState.PRESENT
            slots[slot] = SlotObservation(
                slot=slot,
                state=state,
                detection=winner,
                person_track_id=person.track_id,
            )
        associations.append(
            PersonAssociation(person=person, track_id=person.track_id, slots=slots)
        )

    return AssociationReport(
        associations=tuple(associations),
        unassigned=tuple(unassigned),
        bound=bound,
    )


def relocate_track_id(association: PersonAssociation, track_id: int) -> PersonAssociation:
    """Return ``association`` with its person (and slots) re-keyed to ``track_id``.

    Used by the pipeline when a person arrives without a track ID and is given
    a synthetic one.
    """
    person = with_track_id(association.person, track_id)
    slots = {
        name: SlotObservation(
            slot=obs.slot,
            state=obs.state,
            detection=obs.detection,
            person_track_id=track_id,
        )
        for name, obs in association.slots.items()
    }
    return PersonAssociation(person=person, track_id=track_id, slots=slots)
