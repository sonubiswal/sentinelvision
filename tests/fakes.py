"""Fake ultralytics surface -- no torch, no weights, no network, no camera.

:class:`FakeBoxes` / :class:`FakeResult` / :class:`FakeModel` mirror the parts of
the ultralytics API this project actually touches, including ``.id`` (track IDs)
and ``model.track(...)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FakeBoxes:
    """Mimics ``ultralytics.engine.results.Boxes``."""

    def __init__(
        self,
        xyxy: list[list[float]],
        conf: list[float],
        cls: list[int],
        ids: list[int] | None = None,
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=float)
        self.cls = np.asarray(cls, dtype=float)
        self.id = None if ids is None else np.asarray(ids, dtype=float)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class FakeResult:
    """Mimics a single element of the ``Results`` list."""

    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes


class FakeModel:
    """Minimal stand-in for ``ultralytics.YOLO``.

    ``predict`` and ``track`` return the same boxes; ``track`` is the one that
    carries IDs in a real deployment.  Both call logs are recorded so tests can
    assert *how* the model was queried, not just what came back.
    """

    def __init__(
        self,
        boxes: FakeBoxes | None,
        names: Any = None,
        ids: list[int] | None = None,
        *,
        track_ids: list[int] | None = None,
    ) -> None:
        self._result = FakeResult(boxes)
        self.names = names if names is not None else {0: "person"}
        self.calls: list[dict[str, Any]] = []
        self.track_calls: list[dict[str, Any]] = []
        self._ids = ids
        self._track_ids = track_ids if track_ids is not None else ids

    def _boxes_for(self, tracked: bool) -> FakeResult:
        if not tracked or self._track_ids is None:
            return self._result
        boxes = self._result.boxes
        if boxes is None:
            return self._result
        return FakeResult(
            FakeBoxes(
                xyxy=boxes.xyxy.tolist(),
                conf=boxes.conf.tolist(),
                cls=[int(c) for c in boxes.cls.tolist()],
                ids=list(self._track_ids),
            )
        )

    def predict(self, frame: Any, conf: float = 0.25, iou: float = 0.45, verbose: bool = False):
        """Stateless inference (``Detector.detect``)."""
        self.calls.append({"conf": conf, "iou": iou, "verbose": verbose, "frame": frame})
        return [self._boxes_for(tracked=False)]

    def track(
        self,
        frame: Any,
        conf: float = 0.25,
        iou: float = 0.45,
        persist: bool = True,
        tracker: str | None = None,
        verbose: bool = False,
    ):
        """Tracking inference (``Detector.track``)."""
        self.track_calls.append(
            {
                "conf": conf,
                "iou": iou,
                "persist": persist,
                "tracker": tracker,
                "verbose": verbose,
                "frame": frame,
            }
        )
        return [self._boxes_for(tracked=True)]


class TrackinglessModel(FakeModel):
    """A model with no ``track`` method, for error-path tests."""

    track = None  # type: ignore[assignment]


class FakeCapture:
    """A readable stream of frames that ends after the last one."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self.released = False
        self.read_count = 0

    def read(self):
        self.read_count += 1
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


def blank_frame(height: int = 64, width: int = 64) -> np.ndarray:
    """A black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class ScriptedModel(FakeModel):
    """Returns a different set of boxes for each successive call.

    A real stream changes frame to frame; :class:`FakeModel` cannot express
    that, and without it no temporal test (voting, hysteresis, eviction) is
    meaningful.  The last scripted entry repeats once the script is exhausted,
    and ``ids_script`` may be a single entry to mean "this list every frame".
    """

    def __init__(
        self,
        boxes_script: list[FakeBoxes | None],
        names: Any = None,
        ids_script: list[list[int]] | None = None,
    ) -> None:
        super().__init__(None, names=names)
        if not boxes_script:
            raise ValueError("boxes_script must contain at least one entry")
        self._boxes_script = list(boxes_script)
        self._ids_script = list(ids_script) if ids_script is not None else None
        self._cursor = 0

    @property
    def cursor(self) -> int:
        """How many frames have been served (useful in assertions)."""
        return self._cursor

    def _pop(self) -> FakeResult:
        index = min(self._cursor, len(self._boxes_script) - 1)
        self._cursor += 1
        boxes = self._boxes_script[index]
        if boxes is None:
            return FakeResult(None)
        # Track IDs come from ``ids_script`` when supplied, otherwise from the
        # boxes themselves -- so a script entry can carry its own IDs and the
        # ID list always matches the box count (a mismatch would silently
        # mis-key workers in the pipeline).
        if self._ids_script is not None:
            ids: list[int] | None = self._ids_script[min(index, len(self._ids_script) - 1)]
        elif boxes.id is not None:
            ids = [int(value) for value in boxes.id.tolist()]
        else:
            ids = None
        return FakeResult(
            FakeBoxes(
                xyxy=boxes.xyxy.tolist(),
                conf=boxes.conf.tolist(),
                cls=[int(c) for c in boxes.cls.tolist()],
                ids=ids,
            )
        )

    def predict(self, frame: Any, conf: float = 0.25, iou: float = 0.45, verbose: bool = False):
        self.calls.append({"conf": conf, "iou": iou, "verbose": verbose, "frame": frame})
        return [self._pop()]

    def track(
        self,
        frame: Any,
        conf: float = 0.25,
        iou: float = 0.45,
        persist: bool = True,
        tracker: str | None = None,
        verbose: bool = False,
    ):
        self.track_calls.append(
            {
                "conf": conf,
                "iou": iou,
                "persist": persist,
                "tracker": tracker,
                "verbose": verbose,
                "frame": frame,
            }
        )
        return [self._pop()]


def make_boxes(entries: list[tuple[list[float], float, int]], ids: list[int] | None = None) -> FakeBoxes:
    """Build ``FakeBoxes`` from ``[(xyxy, conf, cls), ...]``."""
    return FakeBoxes(
        xyxy=[entry[0] for entry in entries],
        conf=[entry[1] for entry in entries],
        cls=[entry[2] for entry in entries],
        ids=ids,
    )
