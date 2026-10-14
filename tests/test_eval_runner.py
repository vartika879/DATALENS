"""Tests for the eval runner with an injected fake question runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from datalens.evals.runner import EvalRunner, RunOutcome, format_summary, write_report
from datalens.evals.schema import (
    AnswerKind,
    Difficulty,
    GoldenDataset,
    GoldenQuestion,
)


def _tiny_dataset() -> GoldenDataset:
    return GoldenDataset(
        name="tiny",
        csv_filename="sales.csv",
        description="test",
        questions=[
            GoldenQuestion(
                id="t1",
                question="total?",
                difficulty=Difficulty.EASY,
                expected_kind=AnswerKind.SCALAR,
                expected=2600,
            ),
            GoldenQuestion(
                id="t2",
                question="count?",
                difficulty=Difficulty.EASY,
                expected_kind=AnswerKind.SCALAR,
                expected=8,
            ),
        ],
    )


@pytest.fixture
def sales_csv() -> Path:
    # Reuse the sales fixture from evals/fixtures/
    from datalens.evals.loader import FIXTURES_DIR

    return FIXTURES_DIR / "sales.csv"


def test_runner_with_perfect_fake(sales_csv: Path) -> None:
    def fake(question, df, dataset_id) -> RunOutcome:
        # Return the golden answer for whichever question comes in
        if question.id == "t1":
            return RunOutcome(status="success", actual=2600, retry_count=0, latency_ms=100.0)
        return RunOutcome(status="success", actual=8, retry_count=0, latency_ms=120.0)

    runner = EvalRunner(_tiny_dataset(), sales_csv, question_runner=fake)
    report = runner.run_all()

    assert report.summary.total == 2
    assert report.summary.correct == 2
    assert report.summary.answer_accuracy == 1.0
    assert report.summary.latency_p50_ms == pytest.approx(110.0, rel=0.15)


def test_runner_with_mixed_fake(sales_csv: Path) -> None:
    def fake(question, df, dataset_id) -> RunOutcome:
        if question.id == "t1":
            return RunOutcome(status="success", actual=2600, retry_count=0, latency_ms=100.0)
        return RunOutcome(status="success", actual=999, retry_count=1, latency_ms=200.0)

    runner = EvalRunner(_tiny_dataset(), sales_csv, question_runner=fake)
    report = runner.run_all()

    assert report.summary.correct == 1
    assert report.summary.answer_accuracy == 0.5
    assert report.summary.avg_retries == 0.5


def test_runner_handles_exception(sales_csv: Path) -> None:
    def bad(question, df, dataset_id) -> RunOutcome:
        raise RuntimeError("boom")

    runner = EvalRunner(_tiny_dataset(), sales_csv, question_runner=bad)
    report = runner.run_all()

    assert report.summary.total == 2
    assert report.summary.correct == 0
    assert all(r.execution_status == "error" for r in report.results)


def test_write_report_creates_json(sales_csv: Path, tmp_path: Path) -> None:
    def fake(question, df, dataset_id) -> RunOutcome:
        return RunOutcome(status="success", actual=2600, retry_count=0, latency_ms=100.0)

    runner = EvalRunner(_tiny_dataset(), sales_csv, question_runner=fake)
    report = runner.run_all()
    out = write_report(report, tmp_path)

    assert out.is_file()
    assert out.suffix == ".json"
    assert "tiny" in out.name


def test_format_summary_contains_key_metrics(sales_csv: Path) -> None:
    def fake(question, df, dataset_id) -> RunOutcome:
        return RunOutcome(status="success", actual=2600, retry_count=0, latency_ms=100.0)

    runner = EvalRunner(_tiny_dataset(), sales_csv, question_runner=fake)
    report = runner.run_all()
    text = format_summary(report)

    assert "Eval report" in text
    assert "Total questions" in text
    assert "Latency p50" in text
