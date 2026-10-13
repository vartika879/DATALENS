"""Consistency test: golden JSON must match the reference implementation.

If this test fails, either the CSV changed or a question's reference
expression changed. Run `uv run python evals/build_golden.py` to
regenerate.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from datalens.evals.loader import FIXTURES_DIR, GOLDEN_DIR


def _load_questions_module(name: str):
    import importlib

    return importlib.import_module(f"evals.questions.{name}")


@pytest.mark.parametrize("name", ["sales", "titanic"])
def test_golden_matches_reference(name: str) -> None:
    """Every expected value in the golden JSON must match the reference."""
    golden_path = GOLDEN_DIR / f"{name}.json"
    if not golden_path.is_file():
        pytest.skip(f"{golden_path.name} not generated yet")

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    module = _load_questions_module(name)
    df = pd.read_csv(FIXTURES_DIR / f"{name}.csv")

    by_id = {q["id"]: q for q in golden["questions"]}
    for qdef in module.QUESTIONS:
        qid = qdef["id"]
        assert qid in by_id, f"{qid} missing from golden JSON"

        expected_now = qdef["reference"](df)
        stored = by_id[qid]["expected"]

        # For numeric values, compare loosely. For lists/strings, exact.
        if isinstance(stored, (int, float)) and not isinstance(stored, bool):
            assert float(expected_now) == pytest.approx(float(stored), rel=1e-9), (
                f"{qid}: golden={stored}, reference now={expected_now}"
            )
        else:
            assert expected_now == stored, (
                f"{qid}: golden={stored!r}, reference now={expected_now!r}"
            )
