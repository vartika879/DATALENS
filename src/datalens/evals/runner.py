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
from typing import Literal

import pandas as pd

from datalens.evals.aggregate import compute_summary
from datalens.evals.compare import answers_match
from datalens.evals.report import EvalReport, QuestionResult
from datalens.evals.schema import GoldenDataset, GoldenQuestion
from datalens.pipeline import PipelineResult, run_conversation, run_pipeline
from datalens.registry import register_dataset
from datalens.session import create_session


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
        variant: Literal["auto", "single_shot", "graph"] = "auto",
    ) -> None:
        self.dataset = dataset
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.variant = variant
        self.question_runner = question_runner or self._default_runner

    # --- Default runner -----------------------------------------------------
    def _default_runner(
        self, question: GoldenQuestion, df: pd.DataFrame, dataset_id: str
    ) -> RunOutcome:
        """Run one question.

        Adversarial questions go through the agent graph so that the
        planner can refuse. Other questions use the faster single-shot
        pipeline.
        """
        from datalens.evals.schema import AnswerKind, Difficulty
        from datalens.schema import extract_schema

        schema = extract_schema(self.csv_path)

        # Adversarial → graph pipeline (planner refuses)
        # Variant override: force one mode for all questions.
        if self.variant == "single_shot":
            result = run_pipeline(schema, question.question, df)
            return self._pipeline_result_to_outcome(result)

        if self.variant == "graph":
            return self._run_via_graph(question, schema, df)

        # Auto: adversarial → graph, otherwise single-shot.
        if (
            question.difficulty == Difficulty.ADVERSARIAL
            or question.expected_kind == AnswerKind.REFUSAL
        ):
            return self._run_via_graph(question, schema, df)

        result = run_pipeline(schema, question.question, df)
        return self._pipeline_result_to_outcome(result)

    def _run_via_graph(
        self,
        question: GoldenQuestion,
        schema: object,
        df: pd.DataFrame,
    ) -> RunOutcome:
        """Run a question through the LangGraph agent (planner + nodes)."""
        did = register_dataset(df, schema, csv_path=self.csv_path)  # type: ignore[arg-type]
        sid = create_session()
        result = run_conversation(sid, did, question.question)

        # Extract the "answer" text the agent produced. For refusals, this
        # is the planner's final_answer. For successful runs, it's the
        # summarizer's natural-language answer.
        final = result.get("final_answer") or ""
        status = result.get("execution_status", "success")
        # If planner refused or clarify, execution_status won't be set.
        if not status:
            status = "success"

        # Prefer structured result_data if present (numeric answers).
        actual: object = final
        if result.get("execution_status") == "success" and "result_data" in result:
            actual = result["result_data"]

        return RunOutcome(
            status="success" if status != "error" else "error",
            actual=actual,
            retry_count=result.get("retry_count", 0),
            latency_ms=result.get("total_duration_ms", 0.0),  # type: ignore[arg-type]
        )

    def _pipeline_result_to_outcome(self, result: PipelineResult) -> RunOutcome:
        """Adapt a single-shot PipelineResult to a RunOutcome."""
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
                actual=result.error_message,
                retry_count=0,
                latency_ms=result.total_duration_ms,
                error=result.error_message,
            )

        actual = None
        if result.execution is not None:
            actual = result.execution.result_data
            if result.execution.result_type == "dataframe" and isinstance(actual, dict):
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

    def _call_with_retry(
        self,
        fn: Callable[[], RunOutcome],
        max_retries: int = 3,
        base_delay: float = 5.0,
    ) -> RunOutcome:
        """Call a runner function with exponential backoff on rate limits.

        Detects rate limit errors by inspecting the error message for
        common markers ("429", "rate limit", "too many requests"). Sleeps
        base_delay, 2*base_delay, 4*base_delay between attempts.
        """
        import time as _time

        last_outcome: RunOutcome | None = None
        for attempt in range(max_retries + 1):
            try:
                outcome = fn()
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if (
                    any(m in msg for m in ("429", "rate limit", "too many"))
                    and attempt < max_retries
                ):
                    _time.sleep(base_delay * (2**attempt))
                    continue
                raise
            err = (outcome.error or "").lower()
            if any(m in err for m in ("429", "rate limit", "too many")):
                last_outcome = outcome
                if attempt < max_retries:
                    _time.sleep(base_delay * (2**attempt))
                    continue

            return outcome

        assert last_outcome is not None
        return last_outcome

    def run_question(self, question: GoldenQuestion) -> QuestionResult:
        """Run one golden question and evaluate correctness."""
        try:
            outcome = self._call_with_retry(
                lambda: self.question_runner(question, self.df, self.dataset.name)
            )
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
        delay_seconds: float = 0.0,
    ) -> EvalReport:
        """Run every question and return a complete report.

        Args:
            progress: Optional callback for live progress.
            delay_seconds: Sleep between questions. Useful when hitting
                provider rate limits during ablation runs.
        """
        import time as _time

        results: list[QuestionResult] = []
        total = len(self.dataset.questions)
        for i, q in enumerate(self.dataset.questions, start=1):
            r = self.run_question(q)
            results.append(r)
            if progress:
                progress(i, total, r)
            if delay_seconds > 0 and i < total:
                _time.sleep(delay_seconds)

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
