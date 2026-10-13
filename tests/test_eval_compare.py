"""Tests for the answer comparison logic."""

from __future__ import annotations

from datalens.evals.compare import answers_match
from datalens.evals.schema import AnswerKind

# --- Scalars ----------------------------------------------------------------


def test_scalar_exact_match() -> None:
    ok, reason = answers_match(2600, 2600, kind=AnswerKind.SCALAR)
    assert ok
    assert reason == ""


def test_scalar_int_vs_float() -> None:
    ok, _ = answers_match(2600.0, 2600, kind=AnswerKind.SCALAR)
    assert ok


def test_scalar_within_relative_tolerance() -> None:
    # 57500 vs 57501 → relative diff ~1.7e-5, well under 1%
    ok, _ = answers_match(57501, 57500, kind=AnswerKind.SCALAR)
    assert ok


def test_scalar_outside_relative_tolerance() -> None:
    ok, reason = answers_match(60000, 57500, kind=AnswerKind.SCALAR)
    assert not ok
    assert "numeric mismatch" in reason


def test_scalar_nan_fails() -> None:
    ok, _ = answers_match(float("nan"), 2600, kind=AnswerKind.SCALAR)
    assert not ok


def test_scalar_wrong_type() -> None:
    ok, reason = answers_match("2600", 2600, kind=AnswerKind.SCALAR)
    assert not ok
    assert "expected number" in reason


def test_scalar_bool_is_not_number() -> None:
    """A bool must not silently match int 1."""
    ok, _ = answers_match(True, 1, kind=AnswerKind.SCALAR)
    assert not ok


# --- Strings ----------------------------------------------------------------


def test_string_exact_match() -> None:
    ok, _ = answers_match("Delhi", "Delhi", kind=AnswerKind.STRING)
    assert ok


def test_string_case_insensitive() -> None:
    ok, _ = answers_match("delhi", "DELHI", kind=AnswerKind.STRING)
    assert ok


def test_string_whitespace_stripped() -> None:
    ok, _ = answers_match("  Delhi  ", "Delhi", kind=AnswerKind.STRING)
    assert ok


def test_string_mismatch() -> None:
    ok, reason = answers_match("Mumbai", "Delhi", kind=AnswerKind.STRING)
    assert not ok
    assert "string mismatch" in reason


# --- Lists ------------------------------------------------------------------


def test_list_same_order() -> None:
    ok, _ = answers_match(["a", "b", "c"], ["a", "b", "c"], kind=AnswerKind.LIST)
    assert ok


def test_list_order_independent() -> None:
    ok, _ = answers_match(["c", "a", "b"], ["a", "b", "c"], kind=AnswerKind.LIST)
    assert ok


def test_list_case_and_whitespace_normalized() -> None:
    ok, _ = answers_match([" Delhi ", "MUMBAI"], ["mumbai", "delhi"], kind=AnswerKind.LIST)
    assert ok


def test_list_different_length() -> None:
    ok, _ = answers_match(["a", "b"], ["a", "b", "c"], kind=AnswerKind.LIST)
    assert not ok


def test_list_numeric_elements() -> None:
    ok, _ = answers_match([3, 1, 2], [1, 2, 3], kind=AnswerKind.LIST)
    assert ok


# --- Bools ------------------------------------------------------------------


def test_bool_match() -> None:
    ok, _ = answers_match(True, True, kind=AnswerKind.BOOL)
    assert ok


def test_bool_mismatch() -> None:
    ok, _ = answers_match(True, False, kind=AnswerKind.BOOL)
    assert not ok


# --- Tolerance overrides ----------------------------------------------------


def test_custom_tolerance() -> None:
    # With very tight tolerance, this should fail
    ok, _ = answers_match(57501, 57500, kind=AnswerKind.SCALAR, rel_tolerance=1e-6)
    assert not ok
