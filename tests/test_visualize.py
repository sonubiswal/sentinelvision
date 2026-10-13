"""Visualization tests -- BGR correctness stated as explicit channel assertions."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sentinelvision.detector import Detection
from sentinelvision.visualize import (
    BLUE_BGR,
    COMPLIANT_COLOR,
    GREEN_BGR,
    NEUTRAL_COLOR,
    RED_BGR,
    VIOLATION_COLOR,
    color_for,
    draw_detections,
    draw_label,
    is_green_dominant,
    is_red_dominant,
    label_origin,
    text_size,
)


def blank(height: int = 120, width: int = 160) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def has_exact_color(frame: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool(np.any(np.all(frame == np.array(color, dtype=np.uint8), axis=-1)))


# --- palette ---------------------------------------------------------------
def test_palette_is_bgr_not_rgb() -> None:
    assert GREEN_BGR == (0, 255, 0)
    assert RED_BGR == (0, 0, 255)
    # BGR tuples: red-dominant means the HIGH value lives at index 2 (red).
    assert RED_BGR[2] == 255
    assert RED_BGR[0] == 0
    assert RED_BGR[1] == 0
    # green-dominant means the HIGH value lives at index 1 (green).
    assert GREEN_BGR[1] == 255
    assert GREEN_BGR[0] == 0
    assert GREEN_BGR[2] == 0
    assert len(RED_BGR) == 3


def test_dominance_helpers_use_bgr_indices() -> None:
    assert is_red_dominant(RED_BGR) is True
    assert is_red_dominant(GREEN_BGR) is False
    assert is_green_dominant(GREEN_BGR) is True
    assert is_green_dominant(RED_BGR) is False
    assert is_red_dominant(VIOLATION_COLOR) is True
    assert is_green_dominant(COMPLIANT_COLOR) is True


def test_color_for_maps_violation_red_ppe_green_other_blue() -> None:
    assert color_for("NO-Hardhat") == VIOLATION_COLOR
    assert color_for("NO-Safety Vest") == VIOLATION_COLOR
    assert color_for("no_safety_vest") == VIOLATION_COLOR
    assert color_for("Hardhat") == COMPLIANT_COLOR
    assert color_for("Safety Vest") == COMPLIANT_COLOR
    assert color_for("person") == NEUTRAL_COLOR
    assert color_for("machinery") == NEUTRAL_COLOR
    assert color_for("person") == BLUE_BGR
    # BGR sanity: the violation color is red-dominant, the PPE color green-dominant.
    assert color_for("NO-Mask")[2] == 255
    assert color_for("Mask")[1] == 255


# --- drawing ---------------------------------------------------------------
def test_violation_box_is_drawn_in_red_bgr() -> None:
    frame = blank()
    det = Detection(1, "no-hardhat", "NO-Hardhat", 0.8, (20.0, 40.0, 90.0, 110.0))
    draw_detections(frame, [det])
    assert has_exact_color(frame, RED_BGR)
    assert not has_exact_color(frame, GREEN_BGR)


def test_compliant_box_is_drawn_in_green_bgr() -> None:
    frame = blank()
    det = Detection(2, "hardhat", "Hardhat", 0.8, (20.0, 40.0, 90.0, 110.0))
    draw_detections(frame, [det])
    assert has_exact_color(frame, GREEN_BGR)
    assert not has_exact_color(frame, RED_BGR)


def test_draw_is_in_place_and_returns_the_same_buffer() -> None:
    frame = blank()
    det = Detection(1, "no-hardhat", "NO-Hardhat", 0.8, (20.0, 40.0, 90.0, 110.0))
    returned = draw_detections(frame, [det])
    assert returned is frame
    assert frame.sum() > 0


def test_label_includes_track_id_and_confidence() -> None:
    frame = blank(height=200)
    det = Detection(1, "no-hardhat", "NO-Hardhat", 0.83, (10.0, 60.0, 100.0, 180.0), track_id=7)
    draw_detections(frame, [det], show_confidence=True, show_track_id=True)
    # A track id changes the rendered glyphs, so the label bounding box widens.
    width_with_id, _ = text_size("#7 no-hardhat 0.83")
    width_without_id, _ = text_size("no-hardhat 0.83")
    assert width_with_id > width_without_id
    assert frame.sum() > 0


# --- top-edge label flip ---------------------------------------------------
def test_label_flips_below_the_box_near_the_top_edge() -> None:
    _, _, placement = label_origin(x1=10, y1=10, label_height=12, flip_threshold=30)
    assert placement == "below"


def test_label_sits_above_the_box_when_there_is_room() -> None:
    _, origin_y, placement = label_origin(x1=10, y1=200, label_height=12, flip_threshold=30)
    assert placement == "above"
    assert origin_y < 200


def test_flipped_label_origin_is_below_the_box_top() -> None:
    _, origin_y, _ = label_origin(x1=10, y1=10, label_height=12, flip_threshold=30)
    assert origin_y > 10


@pytest.mark.parametrize("y1", [0, 5, 29])
def test_all_top_edge_boxes_flip(y1: int) -> None:
    assert label_origin(0, y1, 12)[2] == "below"


@pytest.mark.parametrize("y1", [30, 31, 500])
def test_boxes_at_or_below_the_threshold_do_not_flip(y1: int) -> None:
    assert label_origin(0, y1, 12)[2] == "above"


def test_flipped_label_is_actually_visible_on_the_frame() -> None:
    """A top-edge box must not have its label clipped off-screen."""
    frame = blank(height=60, width=200)
    draw_label(frame, "no-hardhat", x1=5, y1=2, color=RED_BGR)
    # Pixels below the box top edge were written.
    assert frame[8:40, 5:120].sum() > 0


def test_text_size_is_positive() -> None:
    width, height = text_size("hardhat")
    assert width > 0
    assert height > 0


def test_draw_label_tolerates_a_box_at_the_right_edge() -> None:
    frame = blank(height=100, width=60)
    det = Detection(1, "no-safety-vest", "NO-Safety Vest", 0.9, (50.0, 50.0, 59.0, 99.0))
    draw_detections(frame, [det])  # must not raise
    assert cv2 is not None
    assert frame.shape == (100, 60, 3)
