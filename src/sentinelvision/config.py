"""Configuration: the SentinelVision class taxonomy and runtime settings.

The taxonomy below is the contract between the model, the tracker, the rule
engine and the audit log.  Everything downstream is derived from it -- there is
no hard-coded class *count* anywhere in the codebase, so adding or removing a
class cannot silently desynchronise the violation logic.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Class taxonomy ---------------------------------------------------------
# Keys are the raw class names as emitted by the model / dataset.  Case and
# spaces are significant in the source data ("NO-Safety Vest" contains a space).
# Values are per-class confidence thresholds: a missed worker is cheap, a false
# violation erodes trust.
CLASS_THRESHOLDS: dict[str, float] = {
    "person": 0.35,
    "Hardhat": 0.50,
    "NO-Hardhat": 0.65,
    "Safety Vest": 0.50,
    "NO-Safety Vest": 0.65,
    "Mask": 0.50,
    "NO-Mask": 0.65,
    "Safety Cone": 0.55,
    "machinery": 0.55,
    "vehicle": 0.55,
}

# Prefix the dataset uses to encode a *missing* PPE item.
VIOLATION_PREFIX = "no-"

# Positive PPE items -- their presence is what makes a worker compliant.
POSITIVE_PPE_CLASSES: tuple[str, ...] = ("hardhat", "safety-vest", "mask")

# Threshold applied to classes we have never seen before (e.g. a model swap).
DEFAULT_FALLBACK_THRESHOLD = 0.25

# --- Phase 2: association -----------------------------------------------
# A PPE detection binds to a person when its centre point falls inside the
# person box, OR IoU(person, ppe) >= ASSOCIATION_IOU_MIN, OR at least
# ASSOCIATION_CONTAINMENT_MIN of the PPE box area lies inside the person box.
ASSOCIATION_IOU_MIN = 0.30
ASSOCIATION_CONTAINMENT_MIN = 0.40

# --- Phase 2: temporal voting -------------------------------------------
# A violation commits when VOTE_THRESHOLD of the last VOTE_WINDOW frames agree
# on the same track ID and the same class.  Resolution needs
# CLEAR_HYSTERESIS consecutive clean frames, so a single good frame cannot
# silently erase a real violation.
VOTE_WINDOW = 10
VOTE_THRESHOLD = 6
CLEAR_HYSTERESIS = 3

# --- Phase 2: tracking --------------------------------------------------
# ByteTrack ships with ultralytics; passing the yaml name is the documented
# way to select it.
TRACKER_CONFIG = "bytetrack.yaml"

# Workers not seen for this many frames are dropped from the registry.
WORKER_MAX_MISSING_FRAMES = 30


class Settings(BaseSettings):
    """Runtime settings, overridable via ``SENTINEL_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    weights: str = "yolov8n.pt"
    device: str = "cpu"
    iou: float = 0.45
    fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD
    class_thresholds: dict[str, float] = Field(default_factory=lambda: dict(CLASS_THRESHOLDS))
    max_frames: int = 0
    log_level: str = "INFO"
    json_logs: bool = False
    stream_fps_guard: float = 60.0

    # Phase 2 -- tracking, association, voting.  Additive: no Phase 1 field
    # changed meaning.
    track: bool = True
    tracker: str = TRACKER_CONFIG
    association_iou_min: float = ASSOCIATION_IOU_MIN
    association_containment_min: float = ASSOCIATION_CONTAINMENT_MIN
    vote_window: int = VOTE_WINDOW
    vote_threshold: int = VOTE_THRESHOLD
    clear_hysteresis: int = CLEAR_HYSTERESIS
    worker_max_missing_frames: int = WORKER_MAX_MISSING_FRAMES
