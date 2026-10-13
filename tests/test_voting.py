"""N-of-M temporal voting: commit, hold, hysteresis, and window handling."""

from __future__ import annotations

from collections import deque

import pytest

from sentinelvision.config import CLEAR_HYSTERESIS, VOTE_THRESHOLD, VOTE_WINDOW
from sentinelvision.tracking import Box
from sentinelvision.voting import (
    DEFAULT_VOTE_CONFIG,
    TemporalVoter,
    VoteConfig,
    count_votes,
    ensure_window,
    update_votes,
)
from sentinelvision.worker import WorkerState

SLUG = "no-hardhat"


def worker(window: int = VOTE_WINDOW, track_id: int = 1) -> WorkerState:
    return WorkerState(track_id=track_id, person_box=Box(0, 0, 10, 20), window=window)


def feed(state: WorkerState, count: int, violations: set[str], config: VoteConfig):
    """Advance ``state`` ``count`` frames, returning the last VoteResult."""
    result = None
    for _ in range(count):
        result = update_votes(state, violations, config)
    return result


# --- defaults and config ---------------------------------------------------
def test_shipped_defaults_are_10_6_3() -> None:
    """The documented defaults, asserted so a silent retune cannot slip through."""
    assert VOTE_WINDOW == 10
    assert VOTE_THRESHOLD == 6
    assert CLEAR_HYSTERESIS == 3
    assert DEFAULT_VOTE_CONFIG.window == 10
    assert DEFAULT_VOTE_CONFIG.threshold == 6
    assert DEFAULT_VOTE_CONFIG.clear_hysteresis == 3
    assert DEFAULT_VOTE_CONFIG.as_dict() == {"window": 10, "threshold": 6, "clear_hysteresis": 3}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window": 0}, "window must be"),
        ({"threshold": 0}, "threshold must be"),
        ({"threshold": 5, "window": 3}, "cannot exceed window"),
        ({"clear_hysteresis": 0}, "clear_hysteresis must be"),
    ],
)
def test_vote_config_rejects_impossible_settings(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        VoteConfig(**kwargs)


# --- counting --------------------------------------------------------------
def test_count_votes_aggregates_across_frames() -> None:
    history = [{"no-hardhat"}, {"no-hardhat", "no-mask"}, {"no-hardhat"}]
    assert count_votes(history) == {"no-hardhat": 3, "no-mask": 1}


def test_count_votes_collapses_duplicates_within_a_frame() -> None:
    """A frame with the same class twice still counts as one frame of evidence."""
    assert count_votes([["no-hardhat", "no-hardhat"]]) == {"no-hardhat": 1}


def test_count_votes_honours_the_window() -> None:
    history = [{"no-mask"}, {"no-hardhat"}, {"no-hardhat"}]
    assert count_votes(history, window=2) == {"no-hardhat": 2}


def test_count_votes_on_empty_history() -> None:
    assert count_votes([]) == {}


# --- commit ----------------------------------------------------------------
def test_violation_commits_exactly_at_k_of_n() -> None:
    state = worker()
    config = VoteConfig(window=10, threshold=6, clear_hysteresis=3)

    assert feed(state, 5, {SLUG}, config).committed == frozenset()
    assert state.violations == set()

    result = feed(state, 1, {SLUG}, config)  # 6th consecutive frame
    assert result.committed == frozenset({SLUG})
    assert result.active == frozenset({SLUG})
    assert state.violations == {SLUG}
    assert result.votes[SLUG] == 6


def test_single_blurred_frame_never_commits() -> None:
    """The whole point: one bad frame is not a violation."""
    state = worker()
    config = VoteConfig(window=10, threshold=6, clear_hysteresis=3)
    result = update_votes(state, {SLUG}, config)
    assert result.committed == frozenset()
    assert result.held == frozenset({SLUG})
    assert state.violations == set()


def test_intermittent_flicker_below_threshold_never_commits() -> None:
    """4 violating frames out of 10: below K=6, so nothing commits."""
    state = worker()
    config = VoteConfig(window=10, threshold=6, clear_hysteresis=3)
    pattern = [{SLUG}, set(), {SLUG}, set(), {SLUG}, set(), set(), {SLUG}, set(), set()]
    for frame in pattern:
        result = update_votes(state, frame, config)
    assert result.committed == frozenset()
    assert result.votes[SLUG] == 4
    assert result.held == frozenset({SLUG})


def test_a_committed_violation_does_not_re_announce_itself() -> None:
    state = worker()
    config = VoteConfig(window=10, threshold=6, clear_hysteresis=3)
    feed(state, 6, {SLUG}, config)
    result = update_votes(state, {SLUG}, config)
    assert result.committed == frozenset()
    assert result.active == frozenset({SLUG})


def test_threshold_one_commits_immediately() -> None:
    state = worker(window=1)
    result = update_votes(state, {SLUG}, VoteConfig(window=1, threshold=1, clear_hysteresis=1))
    assert result.committed == frozenset({SLUG})


# --- resolution hysteresis -------------------------------------------------
def test_resolution_needs_consecutive_clean_frames() -> None:
    state = worker()
    config = VoteConfig(window=10, threshold=6, clear_hysteresis=3)
    feed(state, 6, {SLUG}, config)

    first = update_votes(state, set(), config)
    assert first.cleared == frozenset()
    assert first.active == frozenset({SLUG})

    second = update_votes(state, set(), config)
    assert second.cleared == frozenset()
    assert second.active == frozenset({SLUG})

    third = update_votes(state, set(), config)
    assert third.cleared == frozenset({SLUG})
    assert third.active == frozenset()
    assert state.violations == set()


def test_clean_streak_resets_when_the_violation_reappears() -> None:
    state = worker()
    config = VoteConfig(window=10, threshold=6, clear_hysteresis=3)
    feed(state, 6, {SLUG}, config)
    update_votes(state, set(), config)  # streak 1
    update_votes(state, set(), config)  # streak 2

    update_votes(state, {SLUG}, config)  # observed again -> reset
    assert state.clean_streak[SLUG] == 0

    assert update_votes(state, set(), config).cleared == frozenset()
    assert update_votes(state, set(), config).cleared == frozenset()
    assert update_votes(state, set(), config).cleared == frozenset({SLUG})


def test_cleared_violation_stays_cleared() -> None:
    """Stale window evidence must not resurrect a resolved violation."""
    state = worker()
    config = VoteConfig(window=10, threshold=6, clear_hysteresis=2)
    feed(state, 6, {SLUG}, config)
    update_votes(state, set(), config)
    assert update_votes(state, set(), config).cleared == frozenset({SLUG})

    follow_up = update_votes(state, set(), config)
    assert follow_up.committed == frozenset()
    assert follow_up.active == frozenset()
    assert state.violations == set()


def test_two_violations_are_voted_independently() -> None:
    state = worker()
    config = VoteConfig(window=10, threshold=3, clear_hysteresis=2)
    feed(state, 3, {SLUG}, config)
    assert state.violations == {SLUG}

    result = feed(state, 3, {SLUG, "no-mask"}, config)
    assert result.committed == frozenset({"no-mask"})
    assert result.active == frozenset({SLUG, "no-mask"})


# --- window handling -------------------------------------------------------
def test_ensure_window_realigns_an_existing_deque() -> None:
    state = worker(window=10)
    feed(state, 10, {SLUG}, VoteConfig(window=10, threshold=6, clear_hysteresis=3))
    assert len(state.history) == 10

    ensure_window(state, 3)
    assert state.history.maxlen == 3
    assert len(state.history) == 3
    assert state.window == 3
    assert all(frame == {SLUG} for frame in state.history)


def test_ensure_window_is_a_noop_when_already_correct() -> None:
    state = worker(window=4)
    before = state.history
    ensure_window(state, 4)
    assert state.history is before


def test_update_realigns_the_window_from_the_config() -> None:
    state = worker(window=10)
    for _ in range(8):
        update_votes(state, {SLUG}, VoteConfig(window=4, threshold=2, clear_hysteresis=2))
    assert state.history.maxlen == 4
    assert len(state.history) == 4


def test_shrinking_the_window_can_commit_sooner() -> None:
    state = worker(window=10)
    config = VoteConfig(window=3, threshold=3, clear_hysteresis=2)
    assert feed(state, 3, {SLUG}, config).committed == frozenset({SLUG})


def test_history_is_a_deque_bounded_by_the_window() -> None:
    state = worker(window=5)
    feed(state, 20, {SLUG}, VoteConfig(window=5, threshold=1, clear_hysteresis=1))
    assert isinstance(state.history, deque)
    assert len(state.history) == 5


def test_policy_filtering_is_respected_by_the_voter() -> None:
    """A slot the zone does not require never reaches the voter, so never votes."""
    state = worker()
    config = VoteConfig(window=4, threshold=2, clear_hysteresis=1)
    for _ in range(6):
        result = update_votes(state, set(), config)  # filtered out upstream
    assert result.committed == frozenset()
    assert result.votes == {}
    assert state.violations == set()


# --- TemporalVoter wrapper -------------------------------------------------
def test_temporal_voter_counts_steps_and_uses_its_config() -> None:
    voter = TemporalVoter(VoteConfig(window=3, threshold=2, clear_hysteresis=1))
    state = worker(window=3)
    voter.update(state, {SLUG})
    result = voter.update(state, {SLUG})
    assert voter.steps == 2
    assert result.committed == frozenset({SLUG})
    assert voter.as_dict()["config"]["window"] == 3


def test_temporal_voter_defaults_to_the_shipped_config() -> None:
    assert TemporalVoter().config == DEFAULT_VOTE_CONFIG


def test_temporal_voter_reset_only_clears_its_counter() -> None:
    voter = TemporalVoter(VoteConfig(window=3, threshold=2, clear_hysteresis=1))
    state = worker(window=3)
    voter.update(state, {SLUG})
    voter.update(state, {SLUG})
    voter.reset()
    assert voter.steps == 0
    assert state.violations == {SLUG}  # worker state belongs to the registry


def test_vote_result_as_dict_is_sorted() -> None:
    state = worker(window=3)
    result = update_votes(
        state, {"no-mask", SLUG}, VoteConfig(window=3, threshold=1, clear_hysteresis=1)
    )
    data = result.as_dict()
    assert data["committed"] == ["no-hardhat", "no-mask"]
    assert result.changed is True


def test_vote_result_changed_is_false_while_only_holding() -> None:
    result = update_votes(worker(), {SLUG}, VoteConfig(window=5, threshold=3, clear_hysteresis=1))
    assert result.changed is False
