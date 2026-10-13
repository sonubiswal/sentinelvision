"""Person↔PPE association: the three-state slot logic and its tie-breaks."""

from __future__ import annotations

import pytest

from sentinelvision.association import (
    PPE_SLOTS,
    SlotState,
    associate,
    bind_ppe_to_person,
    is_ppe,
    ppe_slot,
    relocate_track_id,
    violation_slug_for_slot,
)
from sentinelvision.classes import normalize_class_name
from sentinelvision.detector import Detection

NAMES = {
    "person": 0,
    "Hardhat": 1,
    "NO-Hardhat": 2,
    "Safety Vest": 3,
    "NO-Safety Vest": 4,
    "Mask": 5,
    "NO-Mask": 6,
    "machinery": 8,
}


def det(
    class_name: str,
    xyxy: tuple[float, float, float, float],
    confidence: float = 0.9,
    track_id: int | None = None,
) -> Detection:
    return Detection(
        class_id=NAMES.get(class_name, 99),
        class_name=normalize_class_name(class_name),  # Detector normalizes too
        raw_name=class_name,
        confidence=confidence,
        xyxy=xyxy,
        track_id=track_id,
    )


PERSON = (0.0, 0.0, 100.0, 200.0)


# --- slot mapping ----------------------------------------------------------
def test_ppe_slots_are_the_taxonomy_positive_classes() -> None:
    assert PPE_SLOTS == ("hardhat", "safety-vest", "mask")


def test_ppe_slot_maps_positive_and_negative_forms_to_the_same_slot() -> None:
    assert ppe_slot("Hardhat") == "hardhat"
    assert ppe_slot("NO-Hardhat") == "hardhat"
    assert ppe_slot("no-hardhat") == "hardhat"


def test_ppe_slot_handles_the_space_in_no_safety_vest() -> None:
    assert ppe_slot("NO-Safety Vest") == "safety-vest"
    assert ppe_slot("no_safety_vest") == "safety-vest"
    assert ppe_slot("Safety Vest") == "safety-vest"


def test_context_classes_have_no_slot() -> None:
    assert ppe_slot("person") is None
    assert ppe_slot("machinery") is None
    assert ppe_slot("Safety Cone") is None
    assert is_ppe(det("machinery", (0, 0, 5, 5))) is False


def test_violation_slug_for_slot() -> None:
    assert violation_slug_for_slot("safety-vest") == "no-safety-vest"
    assert violation_slug_for_slot("Mask") == "no-mask"


# --- the three states ------------------------------------------------------
def test_person_alone_yields_all_slots_unknown_not_violations() -> None:
    """The core fix: occlusion is UNKNOWN, never 'not wearing a hardhat'."""
    report = associate([det("person", PERSON, track_id=1)], [])
    assert len(report) == 1
    association = report.associations[0]
    assert association.unknown_slots == frozenset(PPE_SLOTS)
    assert association.present_slots == frozenset()
    assert association.violation_slugs == frozenset()
    for slot in PPE_SLOTS:
        assert association.state(slot) is SlotState.UNKNOWN
        assert association.slots[slot].detection is None


def test_bound_positive_detection_is_present() -> None:
    report = associate(
        [det("person", PERSON, track_id=1)],
        [det("Hardhat", (40.0, 10.0, 70.0, 40.0))],
    )
    association = report.associations[0]
    assert association.state("hardhat") is SlotState.PRESENT
    assert association.violation_slugs == frozenset()
    assert association.state("safety-vest") is SlotState.UNKNOWN


def test_bound_negative_detection_is_a_violation() -> None:
    report = associate(
        [det("person", PERSON, track_id=1)],
        [det("NO-Hardhat", (40.0, 10.0, 70.0, 40.0))],
    )
    association = report.associations[0]
    assert association.state("hardhat") is SlotState.VIOLATION
    assert association.violation_slugs == frozenset({"no-hardhat"})
    assert report.violations == frozenset({"no-hardhat"})


def test_all_three_states_coexist_in_one_frame() -> None:
    report = associate(
        [det("person", PERSON, track_id=1)],
        [
            det("Hardhat", (40.0, 10.0, 70.0, 40.0)),  # present
            det("NO-Safety Vest", (30.0, 60.0, 90.0, 130.0)),  # violation
            # mask: nothing bound -> unknown
        ],
    )
    association = report.associations[0]
    assert association.state("hardhat") is SlotState.PRESENT
    assert association.state("safety-vest") is SlotState.VIOLATION
    assert association.state("mask") is SlotState.UNKNOWN
    assert association.violation_slugs == frozenset({"no-safety-vest"})
    assert association.present_slots == frozenset({"hardhat"})
    assert association.unknown_slots == frozenset({"mask"})


def test_slot_state_helpers() -> None:
    report = associate(
        [det("person", PERSON, track_id=1)], [det("NO-Mask", (40.0, 10.0, 70.0, 40.0))]
    )
    observation = report.associations[0].slots["mask"]
    assert observation.is_violation is True
    assert observation.is_present is False
    assert observation.is_unknown is False
    assert observation.confidence == pytest.approx(0.9)
    assert observation.violation_slug == "no-mask"
    assert observation.as_dict()["state"] == "violation"


# --- binding predicates ----------------------------------------------------
def test_binds_when_ppe_centre_is_inside_the_person() -> None:
    person = det("person", PERSON)
    assert bind_ppe_to_person(det("Hardhat", (40.0, 10.0, 70.0, 40.0)), [person]) is person


def test_binds_via_containment_when_centre_escapes() -> None:
    """A hardhat clipped by the person-box edge still binds.

    25x20 box at the right edge: centre x = 102.5 is *outside* the person box,
    and IoU is only 0.019 -- so this can only bind via containment (200/500 =
    0.40, exactly at the configured floor).
    """
    person = det("person", (0.0, 0.0, 100.0, 100.0))
    clipped = det("Hardhat", (90.0, 40.0, 115.0, 60.0))
    assert bind_ppe_to_person(clipped, [person]) is person


def test_binds_via_iou_for_spanning_boxes() -> None:
    person = det("person", (0.0, 0.0, 100.0, 100.0))
    spanning = det("NO-Safety Vest", (10.0, 10.0, 110.0, 110.0))
    assert bind_ppe_to_person(spanning, [person]) is person


def test_does_not_bind_a_distant_ppe_box() -> None:
    """A cone across the yard is not this worker's mask."""
    person = det("person", PERSON)
    assert bind_ppe_to_person(det("Mask", (900.0, 900.0, 920.0, 920.0)), [person]) is None


def test_does_not_bind_when_there_are_no_persons() -> None:
    assert bind_ppe_to_person(det("Hardhat", (10.0, 10.0, 20.0, 20.0)), []) is None


def test_iou_and_containment_thresholds_are_configurable() -> None:
    person = det("person", (0.0, 0.0, 100.0, 100.0))
    # 15x15 box straddling the bottom-right corner: centre y = 102.5 escapes the
    # box, IoU is 0.005, and containment is 50/225 = 0.22.
    ppe = det("Hardhat", (90.0, 95.0, 105.0, 110.0))
    assert bind_ppe_to_person(ppe, [person]) is None  # 0.22 < default 0.40
    assert bind_ppe_to_person(ppe, [person], containment_min=0.2) is person


# --- nearest-person selection ---------------------------------------------
def test_nearest_person_wins_among_overlapping_persons() -> None:
    left = det("person", (0.0, 0.0, 100.0, 100.0), track_id=1)
    right = det("person", (60.0, 0.0, 160.0, 100.0), track_id=2)

    near_left = det("Hardhat", (70.0, 0.0, 80.0, 20.0))  # centre x=75
    assert bind_ppe_to_person(near_left, [left, right]) is left

    near_right = det("Hardhat", (90.0, 0.0, 100.0, 20.0))  # centre x=95
    assert bind_ppe_to_person(near_right, [left, right]) is right


def test_tie_break_is_the_lower_track_id_then_input_order() -> None:
    high = det("person", (0.0, 0.0, 100.0, 100.0), track_id=5)
    low = det("person", (0.0, 0.0, 100.0, 100.0), track_id=2)
    ppe = det("Hardhat", (40.0, 10.0, 50.0, 20.0))
    assert bind_ppe_to_person(ppe, [high, low]) is low

    first = det("person", (0.0, 0.0, 100.0, 100.0))
    second = det("person", (0.0, 0.0, 100.0, 100.0))
    assert bind_ppe_to_person(ppe, [first, second]) is first


def test_multiple_persons_are_all_emitted() -> None:
    report = associate(
        [det("person", PERSON, track_id=1), det("person", (200.0, 0.0, 300.0, 200.0), track_id=2)],
        [det("NO-Hardhat", (40.0, 10.0, 70.0, 40.0))],
    )
    assert len(report) == 2
    assert report.by_track_id()[1].violation_slugs == frozenset({"no-hardhat"})
    assert report.by_track_id()[2].violation_slugs == frozenset()


# --- competing detections and unassigned PPE ------------------------------
def test_positive_detection_wins_a_confidence_tie() -> None:
    """On an exact tie, prefer the compliant reading: a false violation costs more."""
    report = associate(
        [det("person", PERSON, track_id=1)],
        [
            det("Hardhat", (40.0, 10.0, 70.0, 40.0), confidence=0.70),
            det("NO-Hardhat", (42.0, 12.0, 68.0, 38.0), confidence=0.70),
        ],
    )
    assert report.associations[0].state("hardhat") is SlotState.PRESENT


def test_higher_confidence_wins_even_when_it_is_the_violation() -> None:
    report = associate(
        [det("person", PERSON, track_id=1)],
        [
            det("Hardhat", (40.0, 10.0, 70.0, 40.0), confidence=0.55),
            det("NO-Hardhat", (42.0, 12.0, 68.0, 38.0), confidence=0.80),
        ],
    )
    assert report.associations[0].state("hardhat") is SlotState.VIOLATION


def test_ppe_with_no_owner_is_reported_unassigned_not_dropped() -> None:
    report = associate(
        [det("person", PERSON, track_id=1)],
        [det("Hardhat", (40.0, 10.0, 70.0, 40.0)), det("NO-Mask", (900.0, 900.0, 920.0, 920.0))],
    )
    assert report.bound == 1
    assert len(report.unassigned) == 1
    assert report.unassigned[0].class_name == "no-mask"


def test_context_classes_are_never_bound() -> None:
    report = associate(
        [det("person", PERSON, track_id=1)], [det("machinery", (10.0, 10.0, 20.0, 20.0))]
    )
    assert report.bound == 0
    assert len(report.unassigned) == 0
    assert report.associations[0].unknown_slots == frozenset(PPE_SLOTS)


def test_violation_confidence_is_the_mean_of_bound_violations() -> None:
    report = associate(
        [det("person", PERSON, track_id=1)],
        [
            det("NO-Hardhat", (40.0, 10.0, 70.0, 40.0), confidence=0.80),
            det("NO-Mask", (45.0, 15.0, 65.0, 35.0), confidence=0.60),
        ],
    )
    assert report.associations[0].violation_confidence == pytest.approx(0.70)


def test_violation_confidence_is_zero_without_violations() -> None:
    report = associate([det("person", PERSON, track_id=1)], [])
    assert report.associations[0].violation_confidence == pytest.approx(0.0)


# --- relocation ------------------------------------------------------------
def test_relocate_track_id_moves_the_person_and_its_slots() -> None:
    report = associate(
        [det("person", PERSON)], [det("NO-Hardhat", (40.0, 10.0, 70.0, 40.0))]
    )
    association = report.associations[0]
    assert association.track_id is None

    moved = relocate_track_id(association, -1)
    assert moved.track_id == -1
    assert moved.person.track_id == -1
    assert all(obs.person_track_id == -1 for obs in moved.slots.values())
    assert moved.violation_slugs == frozenset({"no-hardhat"})


def test_association_report_is_iterable_and_measurable() -> None:
    report = associate([det("person", PERSON, track_id=1)], [])
    assert list(report) == [report.associations[0]]
    assert report.by_track_id() == {1: report.associations[0]}
    assert report.associations[0].as_dict()["track_id"] == 1
    assert report.associations[0].box.as_list() == list(PERSON)
