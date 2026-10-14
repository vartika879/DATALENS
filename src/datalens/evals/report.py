"""Eval report models.

A single `QuestionResult` is one question's outcome. `EvalReport` bundles
all results plus an aggregate `EvalSummary`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class QuestionResult(BaseModel):
    """Outcome of running one golden question."""

    question_id: str
    question: str
    difficulty: str
    expected_kind: str
    expected: Any
    actual: Any = None
    matches: bool = False
    match_reason: str = ""
    execution_status: str = "error"  # "success" | "error" | "llm_error"
    retry_count: int = 0
    latency_ms: float = 0.0
    error: str | None = None


class EvalSummary(BaseModel):
    """Aggregate metrics over an eval run."""

    total: int
    executed: int  # questions where pipeline returned success or error (not llm_error)
    correct: int
    execution_success_rate: float  # executed / total
    answer_accuracy: float  # correct / total
    correct_when_executed: float  # correct / executed
    avg_retries: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_latency_s: float


class EvalReport(BaseModel):
    """Top-level eval report."""

    dataset_name: str
    run_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary: EvalSummary
    results: list[QuestionResult]


def summarize(results: list[QuestionResult]) -> EvalSummary:
    """Compute aggregate metrics from individual results."""
    from datalens.evals.aggregate import compute_summary

    return compute_summary(results)
