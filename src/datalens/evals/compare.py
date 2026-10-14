"""Answer comparison with tolerance.

Pure functions — no I/O, no network, no side effects. This is the core of
eval scoring: given an expected answer and an actual answer, do they match?

Design decisions:
- Numeric comparison uses BOTH relative and absolute tolerance.
  Relative catches large-number mismatches (57500 vs 57501);
  absolute catches small-number noise (0.0 vs 1e-10).
- Strings are normalized: strip + lowercase. LLMs often return
  "Delhi" vs "delhi" vs " Delhi " for the same answer.
- Lists are compared order-independently after canonicalization.
  [1, 2, 3] matches [3, 1, 2].
"""

from __future__ import annotations

import math
from typing import Any

from datalens.evals.schema import AnswerKind


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_match(
    actual: float,
    expected: float,
    rel_tolerance: float,
    abs_tolerance: float,
) -> bool:
    """Relative + absolute tolerance, robust against zero and NaN."""
    if math.isnan(actual) or math.isnan(expected):
        return False
    diff = abs(actual - expected)
    if diff <= abs_tolerance:
        return True
    denom = max(abs(expected), 1e-12)
    return (diff / denom) <= rel_tolerance


def _canonical_list(value: Any) -> list[Any]:
    """Convert a list-like into a sorted list of comparable primitives."""
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value:
        if isinstance(item, str):
            out.append(item.strip().lower())
        else:
            out.append(item)
    return sorted(out, key=lambda x: (str(type(x)), str(x)))


def answers_match(
    actual: Any,
    expected: Any,
    *,
    kind: AnswerKind,
    rel_tolerance: float = 0.01,
    abs_tolerance: float = 1e-9,
) -> tuple[bool, str]:
    """Compare an actual answer against the golden answer.

    Returns:
        (matches, reason) — reason is "" on success, or a short explanation
        on failure. Used by the runner for failure analysis.
    """
    if kind is AnswerKind.SCALAR:
        if not _is_number(actual) or not _is_number(expected):
            return False, f"expected number, got {type(actual).__name__}"
        if _numbers_match(float(actual), float(expected), rel_tolerance, abs_tolerance):
            return True, ""
        return False, f"numeric mismatch: {actual!r} vs {expected!r}"

    if kind is AnswerKind.STRING:
        if not isinstance(actual, str):
            return False, f"expected string, got {type(actual).__name__}"
        if actual.strip().lower() == str(expected).strip().lower():
            return True, ""
        return False, f"string mismatch: {actual!r} vs {expected!r}"

    if kind is AnswerKind.LIST:
        if not isinstance(actual, list):
            return False, f"expected list, got {type(actual).__name__}"
        got = _canonical_list(actual)
        want = _canonical_list(expected)
        if got == want:
            return True, ""
        return False, f"list mismatch: {got!r} vs {want!r}"
    if kind is AnswerKind.BOOL:
        if not isinstance(actual, bool):
            return False, f"expected bool, got {type(actual).__name__}"
        if actual is expected:
            return True, ""
        return False, f"bool mismatch: {actual!r} vs {expected!r}"

    if kind is AnswerKind.REFUSAL:
        # Adversarial: the agent is expected to refuse. Pass if the actual
        # answer contains any common refusal marker.
        if not isinstance(actual, str):
            return False, f"expected refusal string, got {type(actual).__name__}"
        text = actual.strip().lower()
        markers = (
            "no ",
            "not ",
            "cannot",
            "can't",
            "doesn't",
            "does not",
            "unavailable",
            "not available",
            "not present",
            "not in the dataset",
            "no such column",
            "column not found",
        )
        if any(m in text for m in markers):
            return True, ""
        return False, f"no refusal marker in: {actual!r}"

    return False, f"unsupported answer kind: {kind}"
