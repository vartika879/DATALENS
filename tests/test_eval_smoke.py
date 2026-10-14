"""Smoke test for the end-to-end eval framework.

Uses the real golden dataset (sales.json) but a mocked question runner,
so no LLM calls are made. Verifies:
- Golden dataset loads from disk
- Runner executes all questions
- Aggregation produces correct metrics
- Report is written to disk and is valid JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datalens.evals.loader import FIXTURES_DIR, GOLDEN_DIR, load_golden_dataset
from datalens.evals.runner import EvalRunner, RunOutcome, write_report


@pytest.fixture
def sales_golden_path() -> Path:
    path = GOLDEN_DIR / "sales.json"
    if not path.is_file():
        pytest.skip("sales.json not generated; run evals/build_golden.py")
    return path


def test_full_eval_pipeline_with_mock_runner(
    sales_golden_path: Path,
    tmp_path: Path,
) -> None:
    """Load a real golden dataset, run with a perfect fake, verify report."""
    dataset = load_golden_dataset(sales_golden_path)
    csv_path = FIXTURES_DIR / dataset.csv_filename

    # Every question comes back "correct" by returning the golden value.
    by_id = {q.id: q for q in dataset.questions}

    def perfect_runner(question, df, dataset_id) -> RunOutcome:
        return RunOutcome(
            status="success",
            actual=by_id[question.id].expected,
            retry_count=0,
            latency_ms=42.0,
        )

    runner = EvalRunner(dataset, csv_path, question_runner=perfect_runner)
    report = runner.run_all()

    # All 5 sales questions must be correct.
    assert report.summary.total == len(dataset.questions)
    assert report.summary.correct == report.summary.total
    assert report.summary.answer_accuracy == 1.0
    assert report.summary.avg_retries == 0.0
    assert report.summary.latency_p50_ms == 42.0

    # Report must be writable and re-readable.
    out = write_report(report, tmp_path)
    assert out.is_file()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["dataset_name"] == "sales"
    assert payload["summary"]["correct"] == payload["summary"]["total"]
    assert len(payload["results"]) == len(dataset.questions)


def test_full_eval_pipeline_with_partial_mock(
    sales_golden_path: Path,
    tmp_path: Path,
) -> None:
    """Some questions wrong, some right — verify aggregation is honest."""
    dataset = load_golden_dataset(sales_golden_path)
    csv_path = FIXTURES_DIR / dataset.csv_filename

    by_id = {q.id: q for q in dataset.questions}
    first_id = dataset.questions[0].id

    def half_runner(question, df, dataset_id) -> RunOutcome:
        # First question wrong; rest correct.
        actual = "definitely wrong" if question.id == first_id else by_id[question.id].expected
        return RunOutcome(
            status="success",
            actual=actual,
            retry_count=0,
            latency_ms=10.0,
        )

    runner = EvalRunner(dataset, csv_path, question_runner=half_runner)
    report = runner.run_all()

    assert report.summary.correct == report.summary.total - 1
    assert report.summary.answer_accuracy == pytest.approx(
        (report.summary.total - 1) / report.summary.total
    )
