"""Aggregation helpers for eval results.

Pure functions — no I/O, no LLM. Testable in isolation.
"""

from __future__ import annotations

import statistics

from datalens.evals.report import EvalSummary, QuestionResult


def _percentile(values: list[float], pct: float) -> float:
    """Compute the pct-th percentile of a list. Returns 0.0 if empty."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lower = int(k)
    upper = lower + 1 if lower + 1 < len(sorted_values) else lower
    weight = k - lower
    if upper == lower:
        return sorted_values[lower]
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def compute_summary(results: list[QuestionResult]) -> EvalSummary:
    """Aggregate a list of question results into a summary."""
    total = len(results)
    if total == 0:
        return EvalSummary(
            total=0,
            executed=0,
            correct=0,
            execution_success_rate=0.0,
            answer_accuracy=0.0,
            correct_when_executed=0.0,
            avg_retries=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            total_latency_s=0.0,
        )

    executed = sum(1 for r in results if r.execution_status != "llm_error")
    correct = sum(1 for r in results if r.matches)

    latencies = [r.latency_ms for r in results if r.latency_ms > 0.0]

    return EvalSummary(
        total=total,
        executed=executed,
        correct=correct,
        execution_success_rate=executed / total,
        answer_accuracy=correct / total,
        correct_when_executed=(correct / executed) if executed else 0.0,
        avg_retries=statistics.fmean([r.retry_count for r in results]),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        total_latency_s=sum(latencies) / 1000.0,
    )
