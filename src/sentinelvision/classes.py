"""Class-name normalization and the violation predicate.

This module is the single source of truth for "what counts as a violation".
It is deliberately data-driven: the violation set is derived from the
configured taxonomy rather than from a hard-coded number or list literal.

Why normalization exists
------------------------
The dataset emits ``NO-Safety Vest`` -- uppercase prefix, a space inside the
name.  Meanwhile ``Safety Vest`` is a *compliant* positive-PPE class and
``NO-Safety Vest`` is a violation.  Any naive ``startswith("NO")`` / ``split``
logic breaks on the space; normalization makes the comparison total.
"""

from __future__ import annotations

from collections.abc import Iterable

from .config import CLASS_THRESHOLDS, POSITIVE_PPE_CLASSES, VIOLATION_PREFIX


def normalize_class_name(name: str) -> str:
    """Normalize a raw class name to a stable, comparable slug.

    ``"NO-Safety Vest"`` -> ``"no-safety-vest"``
    ``"no_safety_vest"`` -> ``"no-safety-vest"``
    ``"  NO  Safety   Vest "`` -> ``"no-safety-vest"``

    The transform is idempotent and case-insensitive.
    """
    if name is None:
        raise TypeError("class name must be a string, got None")
    slug = str(name).strip().lower()
    slug = slug.replace("_", " ").replace("-", " ")
    slug = " ".join(slug.split())
    return slug.replace(" ", "-")


#: The canonical raw spellings, exactly as configured.
KNOWN_CLASSES: tuple[str, ...] = tuple(CLASS_THRESHOLDS)

#: normalized slug -> canonical raw spelling.
NORMALIZED_CLASSES: dict[str, str] = {normalize_class_name(c): c for c in KNOWN_CLASSES}

#: Violation classes, *derived* from the taxonomy (never hard-coded).
VIOLATION_CLASSES: tuple[str, ...] = tuple(
    sorted(slug for slug in NORMALIZED_CLASSES if slug.startswith(VIOLATION_PREFIX))
)

#: Positive PPE classes, intersected with the configured taxonomy.
POSITIVE_PPE_CLASSES: tuple[str, ...] = tuple(
    slug for slug in POSITIVE_PPE_CLASSES if slug in NORMALIZED_CLASSES
)

#: How many violation classes the taxonomy currently defines.
VIOLATION_CLASS_COUNT: int = len(VIOLATION_CLASSES)


def is_violation(name: str) -> bool:
    """Return ``True`` when the class denotes a *missing* PPE item.

    Prefix-based, so classes outside the configured taxonomy (e.g. a future
    ``NO-Gloves``) are still treated as violations rather than silently
    ignored.
    """
    return normalize_class_name(name).startswith(VIOLATION_PREFIX)


def is_positive_ppe(name: str) -> bool:
    """Return ``True`` for compliant PPE evidence (Hardhat, Safety Vest, Mask)."""
    return normalize_class_name(name) in POSITIVE_PPE_CLASSES


def is_compliant(name: str) -> bool:
    """Return ``True`` only for positive PPE evidence.

    Non-PPE classes (``person``, ``machinery``, ``vehicle``) are neither
    compliant evidence nor violations -- they are context.
    """
    return is_positive_ppe(name)


def canonical_name(name: str) -> str:
    """Map any spelling of a known class back to its configured spelling.

    Unknown classes are returned in normalized form so they can still be
    logged and compared consistently.
    """
    slug = normalize_class_name(name)
    return NORMALIZED_CLASSES.get(slug, slug)


def resolve_threshold(name: str, thresholds: dict[str, float], fallback: float) -> float:
    """Look up the configured threshold for ``name``, tolerating any spelling."""
    slug = normalize_class_name(name)
    for raw, value in thresholds.items():
        if normalize_class_name(raw) == slug:
            return float(value)
    return float(fallback)


def iter_classes() -> Iterable[tuple[str, float]]:
    """Yield ``(canonical_name, threshold)`` pairs in configured order."""
    for raw in KNOWN_CLASSES:
        yield raw, float(CLASS_THRESHOLDS[raw])
