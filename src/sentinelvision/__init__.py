"""SentinelVision -- workplace safety intelligence platform.

Phase 1 foundation: class taxonomy, class-aware thresholds, a testable YOLO
wrapper, BGR-correct visualization and a CLI.  No model weights, no GPU and no
camera are required to run the test suite.
"""

from __future__ import annotations

from .classes import (
    KNOWN_CLASSES,
    POSITIVE_PPE_CLASSES,
    VIOLATION_CLASSES,
    canonical_name,
    is_compliant,
    is_positive_ppe,
    is_violation,
    normalize_class_name,
)
from .config import CLASS_THRESHOLDS, Settings
from .detector import Detection, Detector

__version__ = "0.1.0"

__all__ = [
    "CLASS_THRESHOLDS",
    "KNOWN_CLASSES",
    "POSITIVE_PPE_CLASSES",
    "VIOLATION_CLASSES",
    "Detection",
    "Detector",
    "Settings",
    "__version__",
    "canonical_name",
    "is_compliant",
    "is_positive_ppe",
    "is_violation",
    "normalize_class_name",
]
