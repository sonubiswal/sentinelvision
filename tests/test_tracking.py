"""Geometry primitives, track-ID plumbing and PPE classification."""

from __future__ import annotations

import pytest

from sentinelvision.classes import normalize_class_name
from sentinelvision.detector import Detection
from sentinelvision.tracking import (
    PERSON_CLASS,
    Box,
    SyntheticTrackIds,
    box_area,
    box_center,
    box_iou,
    center_distance,
    center_inside,
    containment,
    group_by_class,
    is_person,
    max_box_area,
    non_persons,
    normalized_center_distance,
    persons,
    with_track_id,
    with_track_ids,
)


def det(
    class_name: str = "person",
    xyxy: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
    confidence: float = 0.9,
    track_id: int | None = None,
    class_id: int = 0,
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=normalize_class_name(class_name),  # Detector normalizes too
        raw_name=class_name,
        confidence=confidence,
        xyxy=xyxy,
        track_id=track_id,
    )


# --- Box -------------------------------------------------------------------
def test_box_from_xyxy_geometry() -> None:
    box = Box.from_xyxy([10, 20, 30, 60])
    assert box.width == pytest.approx(20.0)
    assert box.height == pytest.approx(40.0)
    assert box.area == pytest.approx(800.0)
    assert box.center == pytest.approx((20.0, 40.0))
    assert box.diagonal == pytest.approx(44.7213595, rel=1e-6)
    assert box.as_tuple() == (10.0, 20.0, 30.0, 60.0)
    assert box.as_list() == [10.0, 20.0, 30.0, 60.0]


def test_box_from_xyxy_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected 4 coordinates"):
        Box.from_xyxy([1, 2, 3])


def test_box_from_detection_matches_detection_coords() -> None:
    assert Box.from_detection(det(xyxy=(1.0, 2.0, 3.0, 4.0))).as_tuple() == (1.0, 2.0, 3.0, 4.0)


def test_box_from_center_round_trips() -> None:
    box = Box.from_center((50.0, 50.0), width=20.0, height=40.0)
    assert box.as_tuple() == (40.0, 30.0, 60.0, 70.0)
    assert box.center == pytest.approx((50.0, 50.0))


def test_functional_helpers_agree_with_methods() -> None:
    box = Box(0, 0, 10, 20)
    assert box_area(box) == pytest.approx(box.area)
    assert box_center(box) == pytest.approx(box.center)


# --- IoU / containment -----------------------------------------------------
def test_iou_of_identical_boxes_is_one() -> None:
    box = Box(0, 0, 10, 10)
    assert box.iou(box) == pytest.approx(1.0)
    assert box_iou(box, Box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert Box(0, 0, 10, 10).iou(Box(20, 20, 30, 30)) == pytest.approx(0.0)


def test_iou_of_edge_touching_boxes_is_zero() -> None:
    """Sharing an edge is not overlapping: zero intersection area."""
    assert Box(0, 0, 10, 10).iou(Box(10, 0, 20, 10)) == pytest.approx(0.0)


def test_iou_known_overlap_value() -> None:
    # intersection 5x5 = 25, union = 100 + 100 - 25 = 175, IoU = 25/175 = 1/7
    assert Box(0, 0, 10, 10).iou(Box(5, 5, 15, 15)) == pytest.approx(1.0 / 7.0)


def test_iou_with_zero_area_box_is_zero() -> None:
    assert Box(5, 5, 5, 5).iou(Box(0, 0, 10, 10)) == pytest.approx(0.0)


def test_intersection_and_union_areas() -> None:
    a, b = Box(0, 0, 10, 10), Box(5, 5, 15, 15)
    assert a.intersection_area(b) == pytest.approx(25.0)
    assert a.union_area(b) == pytest.approx(175.0)


def test_containment_is_one_when_fully_inside() -> None:
    inner = Box(0, 0, 5, 10)
    outer = Box(0, 0, 10, 10)
    assert containment(inner, outer) == pytest.approx(1.0)


def test_containment_is_half_when_half_inside() -> None:
    inner = Box(0, 0, 10, 10)
    outer = Box(0, 0, 5, 10)
    assert containment(inner, outer) == pytest.approx(0.5)


def test_containment_is_zero_when_outside_or_degenerate() -> None:
    assert containment(Box(50, 50, 60, 60), Box(0, 0, 10, 10)) == pytest.approx(0.0)
    assert containment(Box(1, 1, 1, 1), Box(0, 0, 10, 10)) == pytest.approx(0.0)


def test_containment_ignores_outer_size_unlike_iou() -> None:
    """A hardhat inside a person box: tiny IoU, full containment."""
    hardhat = Box(40, 10, 55, 25)
    person = Box(0, 0, 200, 400)
    assert containment(hardhat, person) == pytest.approx(1.0)
    assert person.iou(hardhat) < 0.01


# --- centre helpers --------------------------------------------------------
def test_center_inside_treats_edges_as_inside() -> None:
    box = Box(0, 0, 10, 10)
    assert box.contains_point(0.0, 0.0) is True
    assert box.contains_point(10.0, 10.0) is True
    assert box.contains_point(10.0, 10.0, inclusive=False) is False
    assert box.contains_point(11.0, 5.0) is False


def test_center_inside_helper_uses_inner_centre() -> None:
    assert center_inside(Box(4, 4, 6, 6), Box(0, 0, 10, 10)) is True
    assert center_inside(Box(40, 40, 60, 60), Box(0, 0, 10, 10)) is False


def test_center_distance_is_euclidean() -> None:
    assert center_distance(Box(0, 0, 10, 10), Box(10, 0, 20, 10)) == pytest.approx(10.0)
    assert center_distance(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == pytest.approx(0.0)


def test_normalized_center_distance_is_scale_invariant() -> None:
    """The same 40 px gap is 'far' for small boxes and 'near' for huge ones."""
    small_ref = Box(0, 0, 10, 10)
    huge_ref = Box(0, 0, 1000, 1000)
    a, b = Box(0, 0, 10, 10), Box(40, 0, 50, 10)
    assert normalized_center_distance(small_ref, a, b) > normalized_center_distance(
        huge_ref, a, b
    )


# --- class helpers ---------------------------------------------------------
def test_person_class_constant_matches_the_taxonomy() -> None:
    assert PERSON_CLASS == "person"


def test_is_person_and_partitions() -> None:
    detections = [det("person"), det("no-hardhat", class_id=1), det("machinery", class_id=2)]
    assert is_person(detections[0]) is True
    assert is_person(detections[1]) is False
    assert [d.class_name for d in persons(detections)] == ["person"]
    assert [d.class_name for d in non_persons(detections)] == ["no-hardhat", "machinery"]


def test_group_by_class_preserves_input_order() -> None:
    detections = [
        det("no-hardhat", class_id=1),
        det("person"),
        det("no-hardhat", class_id=1),
        det("hardhat", class_id=2),
    ]
    grouped = group_by_class(detections)
    assert grouped["no-hardhat"] == [detections[0], detections[2]]
    assert grouped["person"] == [detections[1]]
    assert grouped["hardhat"] == [detections[3]]


def test_group_by_class_normalizes_keys() -> None:
    assert list(group_by_class([det("NO-Safety Vest", class_id=1)])) == ["no-safety-vest"]


# --- track IDs -------------------------------------------------------------
def test_with_track_id_returns_a_copy_and_does_not_mutate() -> None:
    original = det()
    enriched = with_track_id(original, 7)
    assert enriched is not original
    assert enriched.track_id == 7
    assert original.track_id is None
    assert enriched.xyxy == original.xyxy
    assert enriched.confidence == original.confidence


def test_with_track_ids_pairs_positionally() -> None:
    detections = [det(), det()]
    enriched = with_track_ids(detections, [1, 2])
    assert [d.track_id for d in enriched] == [1, 2]


def test_synthetic_track_ids_never_collide_with_real_ids() -> None:
    allocator = SyntheticTrackIds()
    ids = [allocator.next() for _ in range(3)]
    assert ids == [-1, -2, -3]
    assert all(value < 0 for value in ids)  # ByteTrack starts at 1
    allocator.reset()
    assert allocator.next() == -1


def test_max_box_area() -> None:
    assert max_box_area([]) == pytest.approx(0.0)
    assert max_box_area([Box(0, 0, 2, 2), Box(0, 0, 10, 10)]) == pytest.approx(100.0)
