"""Class taxonomy tests -- the contract everything downstream depends on."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sentinelvision.classes import (
    KNOWN_CLASSES,
    NORMALIZED_CLASSES,
    VIOLATION_CLASS_COUNT,
    VIOLATION_CLASSES,
    canonical_name,
    is_compliant,
    is_positive_ppe,
    is_violation,
    normalize_class_name,
    resolve_threshold,
)
from sentinelvision.config import CLASS_THRESHOLDS

EXPECTED_VIOLATIONS = ("no-hardhat", "no-mask", "no-safety-vest")


def test_taxonomy_has_ten_classes() -> None:
    assert len(KNOWN_CLASSES) == 10
    assert len(CLASS_THRESHOLDS) == 10


def test_all_ten_classes_round_trip_through_the_normalizer() -> None:
    for raw in KNOWN_CLASSES:
        slug = normalize_class_name(raw)
        assert slug in NORMALIZED_CLASSES
        # Canonical spelling maps back to exactly the configured string.
        assert canonical_name(raw) == raw
        # Normalizing again changes nothing (idempotent).
        assert normalize_class_name(slug) == slug


def test_no_safety_vest_contains_a_space_and_is_still_detected() -> None:
    """The real bug this test guards: the space inside NO-Safety Vest."""
    raw = "NO-Safety Vest"
    assert " " in raw
    assert normalize_class_name(raw) == "no-safety-vest"
    assert normalize_class_name(raw) != normalize_class_name("Safety Vest")
    assert is_violation(raw) is True
    assert canonical_name(raw) == raw


def test_violation_set_is_derived_from_the_taxonomy_not_hardcoded() -> None:
    derived = tuple(sorted(s for s in NORMALIZED_CLASSES if s.startswith("no-")))
    assert derived == VIOLATION_CLASSES
    assert len(VIOLATION_CLASSES) == VIOLATION_CLASS_COUNT
    # The configured taxonomy defines exactly three NO-* violations.
    assert VIOLATION_CLASSES == EXPECTED_VIOLATIONS
    assert VIOLATION_CLASS_COUNT == 3


@pytest.mark.parametrize("name", EXPECTED_VIOLATIONS)
def test_violations_are_flaggged(name: str) -> None:
    assert is_violation(name) is True
    assert is_compliant(name) is False
    assert is_positive_ppe(name) is False


@pytest.mark.parametrize("name", ["Hardhat", "Safety Vest", "Mask"])
def test_positive_ppe_is_compliant_not_a_violation(name: str) -> None:
    assert is_violation(name) is False
    assert is_positive_ppe(name) is True
    assert is_compliant(name) is True


@pytest.mark.parametrize("name", ["person", "Safety Cone", "machinery", "vehicle"])
def test_context_classes_are_neither_violation_nor_compliance(name: str) -> None:
    assert is_violation(name) is False
    assert is_compliant(name) is False


@pytest.mark.parametrize(
    "raw",
    ["no_safety_vest", "NO-SAFETY-VEST", "  no  safety   vest ", "No-Safety Vest"],
)
def test_normalization_is_case_separator_and_whitespace_insensitive(raw: str) -> None:
    assert normalize_class_name(raw) == "no-safety-vest"
    assert is_violation(raw) is True


def test_unseen_violation_class_is_still_a_violation() -> None:
    assert is_violation("NO-Gloves") is True
    assert "no-gloves" not in VIOLATION_CLASSES  # not in the taxonomy, but caught
    assert canonical_name("NO-Gloves") == "no-gloves"


def test_unknown_non_violation_class() -> None:
    assert is_violation("forklift") is False
    assert canonical_name("Forklift") == "forklift"


def test_resolve_threshold_tolerates_any_spelling() -> None:
    assert resolve_threshold("NO-Safety Vest", CLASS_THRESHOLDS, 0.25) == pytest.approx(0.65)
    assert resolve_threshold("no-safety-vest", CLASS_THRESHOLDS, 0.25) == pytest.approx(0.65)
    assert resolve_threshold("NO_SAFETY_VEST", CLASS_THRESHOLDS, 0.25) == pytest.approx(0.65)
    assert resolve_threshold("mystery-class", CLASS_THRESHOLDS, 0.25) == pytest.approx(0.25)


def test_violation_thresholds_are_stricter_than_positive_ppe() -> None:
    for slug in VIOLATION_CLASSES:
        raw = canonical_name(slug)
        assert CLASS_THRESHOLDS[raw] > CLASS_THRESHOLDS["person"]


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1))
def test_normalization_is_idempotent_for_arbitrary_ascii(text: str) -> None:
    once = normalize_class_name(text)
    assert normalize_class_name(once) == once


@given(st.text(alphabet="no-NOSafetyvet _", min_size=1))
def test_normalized_slugs_never_contain_spaces_or_underscores(text: str) -> None:
    slug = normalize_class_name(text)
    assert " " not in slug
    assert "_" not in slug
