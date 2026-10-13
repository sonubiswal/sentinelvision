"""N-of-M temporal voting.

Frame-level detection is noisy: one blurred frame, one reflection, one badly
occluded hardhat and a violation appears.  Committing on a single frame is how
detection-only systems generate 200 alerts a shift that everyone learns to
ignore.

The rule implemented here:

* a violation **commits** when at least ``VOTE_THRESHOLD`` of the last
  ``VOTE_WINDOW`` frames agree -- same track ID, same class;
* it **stays** committed while the evidence persists;
* it **clears** only after ``CLEAR_HYSTERESIS`` *consecutive* clean frames, so
  one good frame cannot silently erase a real violation.

Default ``K=6`` of ``N=10`` was chosen deliberately: at a 25 fps stream that is
0.24 s of agreement before a violation is committed (fast enough to be useful)
while tolerating four bad frames in ten (enough to kill flicker).  It is
configurable, because a 4 fps substation camera and a 30 fps gate camera want
different answers -- see ``docs/failure-modes.md``.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .config import CLEAR_HYSTERESIS, VOTE_THRESHOLD, VOTE_WINDOW
from .worker import WorkerState


@dataclass(frozen=True)
class VoteConfig:
    """Voting parameters.  Validated on construction."""

    window: int = VOTE_WINDOW
    threshold: int = VOTE_THRESHOLD
    clear_hysteresis: int = CLEAR_HYSTERESIS

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {self.threshold}")
        if self.threshold > self.window:
            raise ValueError(
                f"threshold ({self.threshold}) cannot exceed window ({self.window})"
            )
        if self.clear_hysteresis < 1:
            raise ValueError(f"clear_hysteresis must be >= 1, got {self.clear_hysteresis}")

    def as_dict(self) -> dict[str, int]:
        return {
            "window": self.window,
            "threshold": self.threshold,
            "clear_hysteresis": self.clear_hysteresis,
        }


DEFAULT_VOTE_CONFIG = VoteConfig()


@dataclass(frozen=True)
class VoteResult:
    """Outcome of one voting step for one worker."""

    track_id: int
    #: Violations that crossed the threshold *this* step.
    committed: frozenset[str]
    #: Violations that cleared the hysteresis *this* step.
    cleared: frozenset[str]
    #: Every violation committed after this step.
    active: frozenset[str]
    #: Seen this step but still below the threshold.
    held: frozenset[str]
    #: Class -> vote count over the current window.
    votes: dict[str, int]

    @property
    def changed(self) -> bool:
        return bool(self.committed or self.cleared)

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "committed": sorted(self.committed),
            "cleared": sorted(self.cleared),
            "active": sorted(self.active),
            "held": sorted(self.held),
            "votes": dict(sorted(self.votes.items())),
        }


def ensure_window(worker: WorkerState, window: int) -> None:
    """Realign ``worker.history`` to ``window``, preserving the newest frames.

    Config is additive and hot-swappable, so a window change must not silently
    keep using the old bound.
    """
    if worker.history.maxlen == window:
        return
    worker.history = deque(worker.history, maxlen=window)
    worker.window = window


def count_votes(
    history: Iterable[Iterable[str]],
    *,
    window: int | None = None,
) -> dict[str, int]:
    """Count, per violation class, how many recent frames agreed.

    ``history`` is newest-last.  Only the last ``window`` frames are counted.
    """
    frames = list(history)
    if window is not None:
        frames = frames[-window:]
    counter: Counter[str] = Counter()
    for frame in frames:
        counter.update(set(frame))
    return dict(counter)


def update_votes(
    worker: WorkerState,
    frame_violations: Iterable[str],
    config: VoteConfig = DEFAULT_VOTE_CONFIG,
) -> VoteResult:
    """Advance ``worker`` by one frame and report what changed.

    ``frame_violations`` must already be **policy-filtered** (Phase 3 removes
    violations for PPE a worker's zone does not require) so that a
    non-required item never accumulates votes.
    """
    ensure_window(worker, config.window)

    observed = set(frame_violations)
    worker.record_frame(observed)
    votes = count_votes(worker.history, window=config.window)

    committed = {slug for slug, count in votes.items() if count >= config.threshold}
    newly_committed = frozenset(committed - worker.violations)
    worker.violations |= committed

    cleared: set[str] = set()
    for slug in list(worker.violations):
        if slug in observed:
            worker.clean_streak[slug] = 0
            continue
        streak = worker.clean_streak.get(slug, 0) + 1
        worker.clean_streak[slug] = streak
        if streak >= config.clear_hysteresis:
            cleared.add(slug)

    for slug in cleared:
        worker.violations.discard(slug)
        worker.clean_streak.pop(slug, None)
        # Purge the stale evidence too.  Without this, the cleared violation
        # would still sit in the window at or above the threshold and re-commit
        # on the very next frame -- a violation that can never be resolved.
        for frame in worker.history:
            frame.discard(slug)

    held = frozenset(
        slug for slug, count in votes.items() if 0 < count < config.threshold
    )

    return VoteResult(
        track_id=worker.track_id,
        committed=newly_committed,
        cleared=frozenset(cleared),
        active=frozenset(worker.violations),
        held=held,
        votes=votes,
    )


class TemporalVoter:
    """Thin stateful wrapper so callers need not pass a config every frame."""

    def __init__(self, config: VoteConfig | None = None) -> None:
        self.config = config or DEFAULT_VOTE_CONFIG
        self.steps = 0

    def update(
        self, worker: WorkerState, frame_violations: Iterable[str]
    ) -> VoteResult:
        """Advance ``worker`` one frame."""
        self.steps += 1
        return update_votes(worker, frame_violations, self.config)

    def reset(self) -> None:
        """Reset the step counter (worker state belongs to the registry)."""
        self.steps = 0

    def as_dict(self) -> dict[str, Any]:
        return {"steps": self.steps, "config": self.config.as_dict()}
