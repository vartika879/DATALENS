"""Tests for the golden dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datalens.evals.loader import (
    GOLDEN_DIR,
    GoldenDatasetError,
    load_golden_dataset,
)


def test_load_sample_sales_dataset() -> None:
    dataset = load_golden_dataset(GOLDEN_DIR / "sales.json")
    assert dataset.name == "sales"
    assert dataset.csv_filename == "sales.csv"
    assert len(dataset.questions) == 5
    ids = {q.id for q in dataset.questions}
    assert ids == {"sales_q1", "sales_q2", "sales_q3", "sales_q4", "sales_q5"}


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GoldenDatasetError, match="not found"):
        load_golden_dataset(tmp_path / "nope.json")


def test_load_malformed_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(GoldenDatasetError, match="Invalid JSON"):
        load_golden_dataset(bad)


def test_load_invalid_schema_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"name": "x", "csv_filename": "y.csv", "questions": []}),
        encoding="utf-8",
    )
    # Empty questions list violates min_length=1
    with pytest.raises(GoldenDatasetError, match="Invalid golden dataset schema"):
        load_golden_dataset(bad)


def test_load_missing_csv_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "name": "x",
                "csv_filename": "does-not-exist.csv",
                "questions": [
                    {
                        "id": "q1",
                        "question": "?",
                        "difficulty": "easy",
                        "expected_kind": "scalar",
                        "expected": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(GoldenDatasetError, match="CSV referenced"):
        load_golden_dataset(bad)
