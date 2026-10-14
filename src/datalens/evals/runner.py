"""Eval runner — executes golden questions through the pipeline.

The runner is intentionally decoupled from the actual execution:
`run_one` accepts a callable that runs a single question and returns an
`ExecutionResult`. This makes the runner testable with a fake, and lets us
swap between single-shot pipeline (`run_pipeline`) and multi-turn
(`run_conversation`) later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from datalens.evals.aggregate import compute_summary
from datalens.evals.compare import answers_match
from datalens.evals.report import EvalReport, QuestionResult
from datalens.evals.schema import GoldenDataset, GoldenQuestion
from datalens.pipeline import PipelineResult, run_pipeline


@dataclass
class RunOutcome:
    """What a single-question run returns to the runner."""

    status: str  # "success" | "error" | "llm_error"
    actual: object  # the answer value to compare against expected
    retry_count: int
    latency_ms: float
    error: str | None = None


# Type alias for the per-question runner.
QuestionRunner = Callable[[GoldenQuestion, pd.DataFrame, "str"], RunOutcome]


def default_question_runner(
    question: GoldenQuestion, df: pd.DataFrame, dataset_id: str
) -> RunOutcome:
    """Default runner: uses single-shot pipeline, extracts result_data."""

    # Rebuild schema from the CSV via the dataset's path (caller passes path)
    # For simplicity, we accept the DataFrame and pull column info from it.
    # The pipeline requires a DatasetSchema; the caller supplies it via closure.
    raise NotImplementedError  # replaced by closure in EvalRunner


class EvalRunner:
    """Runs a golden dataset through the pipeline and produces a report."""

    def __init__(
        self,
        dataset: GoldenDataset,
        csv_path: Path,
        question_runner: QuestionRunner | None = None,
    ) -> None:
        self.dataset = dataset
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.question_runner = question_runner or self._default_runner

    # --- Default runner -----------------------------------------------------

    def _default_runner(
        self, question: GoldenQuestion, df: pd.DataFrame, dataset_id: str
    ) -> RunOutcome:
        """Run one question through the single-shot pipeline."""
        from datalens.schema import extract_schema

        schema = extract_schema(self.csv_path)
        result: PipelineResult = run_pipeline(schema, question.question, df)

        if result.status == "llm_error":
            return RunOutcome(
                status="llm_error",
                actual=None,
                retry_count=0,
                latency_ms=result.total_duration_ms,
                error=result.error_message,
            )

        if result.status == "error":
            return RunOutcome(
                status="error",
                actual=None,
                retry_count=0,
                latency_ms=result.total_duration_ms,
                error=result.error_message,
            )

        # status == "success"
        actual = None
        if result.execution is not None:
            actual = result.execution.result_data
            # For scalar result, result_data is the sanitized value directly.
            # For strings, same. For lists (from DataFrame rows), handle below.
            if result.execution.result_type == "dataframe" and isinstance(actual, dict):
                # A dataframe result — extract rows for list-like comparisons
                actual = actual.get("rows")
            if result.execution.result_type == "series" and isinstance(actual, dict):
                actual = actual.get("values")

        return RunOutcome(
            status="success",
            actual=actual,
            retry_count=0,
            latency_ms=result.total_duration_ms,
        )

    # --- Public API ---------------------------------------------------------

    def run_question(self, question: GoldenQuestion) -> QuestionResult:
        """Run one golden question and evaluate correctness."""
        try:
            outcome = self.question_runner(question, self.df, self.dataset.name)
        except Exception as exc:  # noqa: BLE001 — surface as failure
            return QuestionResult(
                question_id=question.id,
                question=question.question,
                difficulty=question.difficulty.value,
                expected_kind=question.expected_kind.value,
                expected=question.expected,
                matches=False,
                match_reason=f"runner raised: {exc}",
                execution_status="error",
                error=str(exc),
            )

        matches = False
        reason = ""
        if outcome.actual is not None:
            matches, reason = answers_match(
                outcome.actual,
                question.expected,
                kind=question.expected_kind,
                rel_tolerance=question.rel_tolerance,
                abs_tolerance=question.abs_tolerance,
            )
        elif outcome.status != "success":
            reason = f"no answer (status={outcome.status})"

        return QuestionResult(
            question_id=question.id,
            question=question.question,
            difficulty=question.difficulty.value,
            expected_kind=question.expected_kind.value,
            expected=question.expected,
            actual=outcome.actual,
            matches=matches,
            match_reason=reason,
            execution_status=outcome.status,
            retry_count=outcome.retry_count,
            latency_ms=outcome.latency_ms,
            error=outcome.error,
        )

    def run_all(
        self,
        progress: Callable[[int, int, QuestionResult], None] | None = None,
    ) -> EvalReport:
        """Run every question and return a complete report."""
        results: list[QuestionResult] = []
        total = len(self.dataset.questions)
        for i, q in enumerate(self.dataset.questions, start=1):
            r = self.run_question(q)
            results.append(r)
            if progress:
                progress(i, total, r)

        return EvalReport(
            dataset_name=self.dataset.name,
            summary=compute_summary(results),
            results=results,
        )


def write_report(report: EvalReport, output_dir: Path) -> Path:
    """Persist a report as JSON in `output_dir` with a timestamped name."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report.run_at.strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{report.dataset_name}_{timestamp}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def format_summary(report: EvalReport) -> str:
    """Human-readable summary string."""
    s = report.summary
    return (
        f"\n=== Eval report: {report.dataset_name} ===\n"
        f"  Total questions      : {s.total}\n"
        f"  Executed             : {s.executed}  ({s.execution_success_rate:.1%})\n"
        f"  Correct              : {s.correct}  ({s.answer_accuracy:.1%})\n"
        f"  Correct when executed: {s.correct_when_executed:.1%}\n"
        f"  Avg retries          : {s.avg_retries:.2f}\n"
        f"  Latency p50 / p95    : {s.latency_p50_ms:.0f} ms / {s.latency_p95_ms:.0f} ms\n"
        f"  Total latency        : {s.total_latency_s:.1f} s\n"
    )
