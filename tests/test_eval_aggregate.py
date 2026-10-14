"""Tests for eval aggregation."""

from __future__ import annotations

import pytest

from datalens.evals.aggregate import _percentile, compute_summary
from datalens.evals.report import QuestionResult


def _qr(
    matches: bool, status: str = "success", latency: float = 100.0, retries: int = 0
) -> QuestionResult:
    return QuestionResult(
        question_id="q",
        question="q?",
        difficulty="easy",
        expected_kind="scalar",
        expected=1,
        actual=1 if matches else 2,
        matches=matches,
        execution_status=status,
        retry_count=retries,
        latency_ms=latency,
    )


def test_percentile_empty() -> None:
    assert _percentile([], 50) == 0.0


def test_percentile_single() -> None:
    assert _percentile([100.0], 95) == 100.0


def test_percentile_p50() -> None:
    assert _percentile([10, 20, 30, 40, 50], 50) == 30.0


def test_percentile_p95() -> None:
    # 5 values → 95th percentile near the max
    assert _percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0, rel=0.05)


def test_compute_summary_empty() -> None:
    s = compute_summary([])
    assert s.total == 0
    assert s.answer_accuracy == 0.0


def test_compute_summary_mixed() -> None:
    results = [
        _qr(matches=True, latency=100),
        _qr(matches=True, latency=200),
        _qr(matches=False, latency=150),
        _qr(matches=False, status="error", latency=50),
        _qr(matches=False, status="llm_error", latency=0),
    ]
    s = compute_summary(results)
    assert s.total == 5
    assert s.executed == 4  # 5 total - 1 llm_error
    assert s.correct == 2
    assert s.answer_accuracy == pytest.approx(0.4)
    assert s.execution_success_rate == pytest.approx(0.8)
    assert s.correct_when_executed == pytest.approx(0.5)
    assert s.total_latency_s == pytest.approx(0.5)


def test_compute_summary_all_correct() -> None:
    results = [_qr(matches=True) for _ in range(3)]
    s = compute_summary(results)
    assert s.answer_accuracy == 1.0
    assert s.correct_when_executed == 1.0
