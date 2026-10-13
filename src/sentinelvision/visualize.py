"""BGR-correct drawing helpers.

OpenCV works in **BGR** order, not RGB.  Getting this wrong is the single most
common "my violation boxes are blue" bug, so the palette below is defined once,
as explicit BGR tuples, and asserted in the test suite:

* ``GREEN_BGR = (0, 255, 0)``  -> green-dominant: index 0 (blue) is 0
* ``RED_BGR   = (0, 0, 255)``  -> red-dominant: index 2 (red) is 255

Labels near the top edge of the frame are flipped *below* the box so they are
never clipped off-screen.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .classes import is_positive_ppe, is_violation, normalize_class_name
from .detector import Detection

# --- Palette (OpenCV BGR order: blue, green, red) ---------------------------
GREEN_BGR: tuple[int, int, int] = (0, 255, 0)
RED_BGR: tuple[int, int, int] = (0, 0, 255)
AMBER_BGR: tuple[int, int, int] = (0, 191, 255)
BLUE_BGR: tuple[int, int, int] = (255, 128, 0)
GRAY_BGR: tuple[int, int, int] = (160, 160, 160)

VIOLATION_COLOR: tuple[int, int, int] = RED_BGR
COMPLIANT_COLOR: tuple[int, int, int] = GREEN_BGR
NEUTRAL_COLOR: tuple[int, int, int] = BLUE_BGR

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
BOX_THICKNESS = 2
PADDING = 4
LABEL_FLIP_THRESHOLD = 30


def color_for(name: str) -> tuple[int, int, int]:
    """Violation -> red, positive PPE -> green, everything else -> blue."""
    slug = normalize_class_name(name)
    if is_violation(slug):
        return VIOLATION_COLOR
    if is_positive_ppe(slug):
        return COMPLIANT_COLOR
    return NEUTRAL_COLOR


def is_red_dominant(color: tuple[int, int, int]) -> bool:
    """True when the red channel (BGR index 2) is the strongest channel."""
    blue, green, red = color[:3]
    return red > blue and red >= green


def is_green_dominant(color: tuple[int, int, int]) -> bool:
    """True when the green channel (BGR index 1) is the strongest channel."""
    blue, green, red = color[:3]
    return green > blue and green >= red


def text_size(text: str) -> tuple[int, int]:
    """Return ``(width, height)`` of ``text`` in pixels for the default font."""
    (width, height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    return int(width), int(height)


def label_origin(
    x1: float,
    y1: float,
    label_height: int,
    *,
    flip_threshold: int = LABEL_FLIP_THRESHOLD,
    padding: int = PADDING,
) -> tuple[int, int, str]:
    """Compute the text baseline origin for a box label.

    Returns ``(origin_x, origin_y, placement)`` where ``placement`` is
    ``"above"`` (normal) or ``"below"`` (flipped because the box hugs the top
    edge of the frame).
    """
    if y1 < flip_threshold:
        return int(x1), int(y1 + label_height + padding), "below"
    return int(x1), int(y1 - padding), "above"


def draw_label(
    frame: np.ndarray,
    text: str,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw ``text`` on a filled background block anchored at a box corner."""
    label_w, label_h = text_size(text)
    origin_x, origin_y, placement = label_origin(x1, y1, label_h)

    offset = label_h + PADDING if placement == "below" else label_h
    top = origin_y - offset

    bottom = top + label_h + 2 * PADDING
    left = origin_x
    right = origin_x + label_w + 2 * PADDING

    frame_h, frame_w = frame.shape[:2]
    top = max(0, top)
    bottom = min(frame_h, bottom)
    right = min(frame_w, right)

    cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
    cv2.putText(
        frame,
        text,
        (origin_x + PADDING, origin_y),
        FONT,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return frame


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    show_confidence: bool = True,
    show_track_id: bool = True,
) -> np.ndarray:
    """Draw every detection in place and return the frame."""
    for det in detections:
        color = color_for(det.class_name)
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        label = det.class_name
        if show_confidence:
            label = f"{label} {det.confidence:.2f}"
        if show_track_id and det.track_id is not None:
            label = f"#{det.track_id} {label}"
        draw_label(frame, label, x1, y1, color)
    return frame
