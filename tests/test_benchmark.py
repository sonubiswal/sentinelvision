"""Tests for the benchmark metric math (weight-free: no video, no labels, no model)."""

from __future__ import annotations

import json

import pytest
from benchmarks.evaluate_video import (
    TBD,
    FrameLabel,
    Score,
    build_parser,
    load_labels,
    safe_divide,
    score_events,
    self_test,
)

from sentinelvision.events import ViolationEvent
from sentinelvision.tracking import Box


def event(
    frame_index: int,
    violations: frozenset[str] = frozenset({"no-hardhat"}),
    track_id: int = 1,
) -> ViolationEvent:
    return ViolationEvent(
        track_id=track_id,
        worker_box=Box(0, 0, 10, 20),
        violations=violations,
        frame_index=frame_index,
        timestamp=0.0,
        confidence=0.9,
    )


def label(
    frame_index: int, violations: frozenset[str] = frozenset({"no-hardhat"}), track_id: int | None = None
) -> FrameLabel:
    return FrameLabel(frame_index=frame_index, violations=violations, track_id=track_id)


# --- arithmetic ------------------------------------------------------------
def test_safe_divide_handles_a_zero_denominator() -> None:
    assert safe_divide(3, 0) == 0.0
    assert safe_divide(0, 0) == 0.0
    assert safe_divide(3, 4) == pytest.approx(0.75)


def test_score_from_known_counts() -> None:
    score = Score(true_positives=8, false_positives=2, false_negatives=2)
    assert score.precision == pytest.approx(0.8)
    assert score.recall == pytest.approx(0.8)
    assert score.f1 == pytest.approx(0.8)


def test_empty_score_is_all_zeros() -> None:
    score = Score()
    assert (score.precision, score.recall, score.f1) == (0.0, 0.0, 0.0)
    assert score.as_dict() == {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_perfect_and_worthless_scores() -> None:
    perfect = Score(5, 0, 0)
    assert (perfect.precision, perfect.recall, perfect.f1) == (1.0, 1.0, 1.0)
    noise = Score(0, 3, 3)
    assert (noise.precision, noise.recall, noise.f1) == (0.0, 0.0, 0.0)


def test_false_positive_rate_is_per_frame() -> None:
    assert Score(1, 4, 0).false_positive_rate(100) == pytest.approx(0.04)
    assert Score(1, 0, 0).false_positive_rate(0) == 0.0  # no division by zero


def test_as_dict_can_include_the_frame_count() -> None:
    data = Score(1, 2, 3).as_dict(frames=50)
    assert data["frames"] == 50
    assert data["false_positive_rate"] == pytest.approx(0.04)


# --- matching --------------------------------------------------------------
def test_exact_frame_match_is_a_true_positive() -> None:
    score = score_events([event(5)], [label(5)])
    assert score == Score(true_positives=1, false_positives=0, false_negatives=0)


def test_match_inside_the_window() -> None:
    assert score_events([event(7)], [label(5, track_id=1)], window=3).true_positives == 1
    assert score_events([event(7)], [label(5, track_id=1)], window=1).true_positives == 0


def test_event_outside_the_window_is_a_false_positive() -> None:
    score = score_events([event(20)], [label(5)], window=3)
    assert score.true_positives == 0
    assert score.false_positives == 1
    assert score.false_negatives == 1


def test_unmatched_label_is_a_false_negative() -> None:
    score = score_events([], [label(1), label(2)])
    assert score.false_negatives == 2
    assert score.false_positives == 0


def test_disjoint_violation_classes_do_not_match() -> None:
    score = score_events([event(5, frozenset({"no-mask"}))], [label(5, frozenset({"no-hardhat"}))])
    assert score.true_positives == 0
    assert score.false_positives == 1
    assert score.false_negatives == 1


def test_overlapping_violation_classes_match() -> None:
    score = score_events(
        [event(5, frozenset({"no-hardhat", "no-mask"}))], [label(5, frozenset({"no-hardhat"}))]
    )
    assert score.true_positives == 1


def test_label_track_id_is_enforced_when_present() -> None:
    wrong = score_events([event(5, track_id=3)], [label(5, track_id=2)])
    assert wrong.true_positives == 0

    right = score_events([event(5, track_id=2)], [label(5, track_id=2)])
    assert right.true_positives == 1


def test_a_label_is_consumed_only_once() -> None:
    score = score_events([event(5), event(5)], [label(5)])
    assert score.true_positives == 1
    assert score.false_positives == 1
    assert score.false_negatives == 0


def test_two_labels_in_one_frame_can_both_match() -> None:
    score = score_events([event(5, track_id=1), event(5, track_id=2)], [label(5, track_id=1), label(5, track_id=2)])
    assert score.true_positives == 2
    assert score.false_negatives == 0


def test_one_event_matching_the_earlier_label_wins_deterministically() -> None:
    """-offset is tried before +offset, so the match does not depend on set order."""
    labels = [label(4), label(6)]
    score = score_events([event(5)], labels, window=1)
    assert score.true_positives == 1


# --- labels file -----------------------------------------------------------
def test_load_labels_from_a_json_list(tmp_path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            [
                {"frame_index": 0, "violations": ["no-hardhat"]},
                {"frame_index": 9, "violations": ["no-mask"], "track_id": 4},
            ]
        ),
        encoding="utf-8",
    )
    labels = load_labels(path)
    assert labels[0] == FrameLabel(0, frozenset({"no-hardhat"}), None)
    assert labels[1] == FrameLabel(9, frozenset({"no-mask"}), 4)


def test_load_labels_accepts_a_wrapped_object(tmp_path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": [{"frame_index": 1, "violations": []}]}), encoding="utf-8")
    assert len(load_labels(path)) == 1


def test_load_labels_rejects_a_non_list(tmp_path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps("nope"), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON list"):
        load_labels(path)


def test_frame_label_from_dict_handles_missing_fields() -> None:
    assert FrameLabel.from_dict({"frame_index": 3}) == FrameLabel(3, frozenset(), None)


# --- self-test / TBD discipline --------------------------------------------
def test_self_test_covers_the_degenerate_cases() -> None:
    report = self_test()
    assert report["tp0_fp0_fn0"]["f1"] == 0.0
    assert report["tp8_fp2_fn2"]["f1"] == pytest.approx(0.8)
    assert report["tp5_fp0_fn0"]["precision"] == 1.0


def test_tbd_placeholder_is_literal() -> None:
    """Guards the no-fabricated-numbers rule: the marker is the string TBD."""
    assert TBD == "TBD"


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.video is None
    assert args.labels is None
    assert args.window == 3
    assert args.self_test is False
    assert args.out == "benchmarks/results.json"


def test_main_refuses_to_print_a_score_without_labels(capsys) -> None:
    from benchmarks.evaluate_video import main

    assert main(["--video", "clip.mp4"]) == 2
    out = capsys.readouterr().out
    assert TBD in out
    assert "violation F1" in out


def test_main_self_test_exits_zero(capsys) -> None:
    from benchmarks.evaluate_video import main

    assert main(["--self-test"]) == 0
    assert "tp8_fp2_fn2" in capsys.readouterr().out
