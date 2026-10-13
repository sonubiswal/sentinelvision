"""Inference wrapper.

Design notes
------------
* ``ultralytics`` is imported **lazily** inside :meth:`Detector.load`.  That
  keeps the test suite (and CI lint job) independent of torch, of downloaded
  weights, of a GPU and of a camera.
* The model is injectable, so tests pass a fake object exposing ``names`` and
  ``predict(...)`` -- the same surface ``ultralytics.YOLO`` exposes.
* Per-class thresholds are applied here, after the model call, which is why
  the model is always queried at the *lowest* threshold in the taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .classes import is_violation as _is_violation
from .classes import normalize_class_name, resolve_threshold
from .config import CLASS_THRESHOLDS, DEFAULT_FALLBACK_THRESHOLD, TRACKER_CONFIG

# --- Model-language bridge ---------------------------------------------
# Some YOLO weights trained on the "Construction Site Safety" dataset emit
# Portuguese class names.  Translate them to the canonical English taxonomy
# so the rest of the pipeline (and classes.py normalization) works unchanged.
_PORTUGUESE_TO_ENGLISH: dict[str, str] = {
    "Capacete": "Hardhat",
    "M\u00e1scara": "Mask",
    "Sem Capacete": "NO-Hardhat",
    "Sem M\u00e1scara": "NO-Mask",
    "Sem Colete": "NO-Safety Vest",
    "Pessoa": "person",
    "Cone de Sinaliza\u00e7\u00e3o": "Safety Cone",
    "Colete Refletor": "Safety Vest",
    "Maquinaria": "machinery",
    "Ve\u00edculo": "vehicle",
}


@dataclass(frozen=True)
class Detection:
    """A single detection, already filtered by its class-aware threshold."""

    class_id: int
    class_name: str  # normalized slug
    raw_name: str  # exactly as reported by the model
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None

    @property
    def x1(self) -> float:
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        return self.xyxy[3]

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
    def is_violation(self) -> bool:
        return _is_violation(self.class_name)


class Detector:
    """Wraps a YOLO-like model and applies class-aware confidence thresholds."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        weights: str = "yolov8n.pt",
        device: str = "cpu",
        thresholds: dict[str, float] | None = None,
        fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
        tracker: str = TRACKER_CONFIG,
    ) -> None:
        self._model = model
        self.weights = weights
        self.device = device
        self.thresholds = dict(thresholds if thresholds is not None else CLASS_THRESHOLDS)
        self.fallback_threshold = float(fallback_threshold)
        self.tracker = tracker

    # -- construction ---------------------------------------------------
    @classmethod
    def load(
        cls,
        weights: str = "yolov8n.pt",
        *,
        device: str = "cpu",
        thresholds: dict[str, float] | None = None,
        tracker: str = TRACKER_CONFIG,
    ) -> Detector:
        """Load real weights.  Requires the optional ``vision`` extra."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "ultralytics is required to load model weights. "
                'Install it with: pip install -e ".[vision]"'
            ) from exc
        model = YOLO(weights)
        return cls(
            model, weights=weights, device=device, thresholds=thresholds, tracker=tracker
        )

    # -- introspection --------------------------------------------------
    @property
    def model(self) -> Any | None:
        return self._model

    @property
    def names(self) -> dict[int, str]:
        """Model class-id -> name mapping, tolerant of list *and* dict.

        Portuguese names emitted by some PPE-trained checkpoints are
        translated to the canonical English taxonomy on the way out.
        """
        raw = getattr(self._model, "names", {}) or {}
        if isinstance(raw, dict):
            mapping = {int(k): str(v) for k, v in raw.items()}
        else:
            mapping = {i: str(v) for i, v in enumerate(raw)}
        return {k: _PORTUGUESE_TO_ENGLISH.get(v, v) for k, v in mapping.items()}

    def name_for(self, class_id: int) -> str:
        return self.names.get(int(class_id), f"class-{class_id}")

    def threshold_for(self, name: str) -> float:
        return resolve_threshold(name, self.thresholds, self.fallback_threshold)

    # -- inference ------------------------------------------------------
    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Detector has no model. Use Detector.load(...) first.")
        return self._model

    def _query_conf(self, conf: float | None) -> float:
        """Query threshold: the lowest configured threshold, unless overridden.

        Per-class filtering happens *after* the model call, so the query must
        never be stricter than the loosest class threshold.
        """
        if conf is not None:
            return float(conf)
        return min([*self.thresholds.values(), self.fallback_threshold])

    def detect(
        self,
        frame: np.ndarray,
        *,
        conf: float | None = None,
        iou: float = 0.45,
    ) -> list[Detection]:
        """Run stateless inference and return threshold-filtered detections.

        ``conf`` overrides the query threshold sent to the model; per-class
        filtering is still applied afterwards.
        """
        model = self._require_model()
        if frame is None:
            raise ValueError("frame must be a numpy array, got None")
        results = model.predict(
            frame, conf=self._query_conf(conf), iou=iou, verbose=False
        )
        return self._parse(results)

    def track(
        self,
        frame: np.ndarray,
        *,
        conf: float | None = None,
        iou: float = 0.45,
        persist: bool = True,
        tracker: str | None = None,
    ) -> list[Detection]:
        """Run inference *with tracking* and populate ``Detection.track_id``.

        Uses ByteTrack (``tracker="bytetrack.yaml"``).  ``persist=True`` keeps
        trajectory state alive across successive calls, which is what makes
        track IDs stable frame to frame.
        """
        model = self._require_model()
        if frame is None:
            raise ValueError("frame must be a numpy array, got None")
        if not hasattr(model, "track"):
            raise RuntimeError(
                "model does not support tracking; use detect() and inject track IDs"
            )
        results = model.track(
            frame,
            conf=self._query_conf(conf),
            iou=iou,
            persist=persist,
            tracker=tracker or self.tracker,
            verbose=False,
        )
        return self._parse(results)

    def reset_tracker(self) -> None:
        """Drop track state between independent video files."""
        predictor = getattr(self._model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        if trackers:  # pragma: no cover - depends on ultralytics internals
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if reset is not None:
                    reset()

    def _parse(self, results: Any) -> list[Detection]:
        """Convert ultralytics ``Results`` into threshold-filtered Detections."""
        detections: list[Detection] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            ids = getattr(boxes, "id", None)
            for index in range(len(boxes)):
                class_id = int(boxes.cls[index])
                raw_name = self.name_for(class_id)
                confidence = float(boxes.conf[index])
                if confidence < self.threshold_for(raw_name):
                    continue
                coords = tuple(float(v) for v in np.asarray(boxes.xyxy[index]).ravel()[:4])
                track_id = int(ids[index]) if ids is not None else None
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=normalize_class_name(raw_name),
                        raw_name=raw_name,
                        confidence=confidence,
                        xyxy=(coords[0], coords[1], coords[2], coords[3]),
                        track_id=track_id,
                    )
                )
        return detections