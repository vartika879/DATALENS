"""Tests for the end-to-end pipeline (LLM mocked)."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from datalens.llm import GeneratedCode, LLMError
from datalens.pipeline import run_pipeline
from datalens.schema import ColumnInfo, DatasetSchema


@pytest.fixture
def schema() -> DatasetSchema:
    return DatasetSchema(
        file_name="x.csv",
        row_count=3,
        column_count=2,
        columns=[
            ColumnInfo(name="city", dtype="object", null_count=0, null_percentage=0.0),
            ColumnInfo(name="sales", dtype="int64", null_count=0, null_percentage=0.0),
        ],
        sample_rows=[{"city": "A", "sales": 10}],
    )


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"city": ["A", "B", "C"], "sales": [10, 20, 30]})


def _generated(code: str) -> GeneratedCode:
    return GeneratedCode(code=code, explanation="test", self_reported_confidence=0.9)


@patch("datalens.pipeline.generate_code")
def test_pipeline_success(mock_gen, schema, df) -> None:
    mock_gen.return_value = _generated("result = df['sales'].sum()")
    result = run_pipeline(schema, "total?", df)

    assert result.status == "success"
    assert result.execution is not None
    assert result.execution.result_data == 60
    assert result.error_message is None


@patch("datalens.pipeline.generate_code")
def test_pipeline_llm_error(mock_gen, schema, df) -> None:
    mock_gen.side_effect = LLMError("api down")
    result = run_pipeline(schema, "total?", df)

    assert result.status == "llm_error"
    assert "api down" in (result.error_message or "")
    assert result.generated is None
    assert result.execution is None


@patch("datalens.pipeline.generate_code")
def test_pipeline_execution_error(mock_gen, schema, df) -> None:
    mock_gen.return_value = _generated("result = df['nope'].sum()")
    result = run_pipeline(schema, "bad column", df)

    assert result.status == "error"
    assert result.generated is not None
    assert result.execution is not None
    assert result.execution.status == "error"
    assert "nope" in (result.error_message or "")


@patch("datalens.pipeline.generate_code")
def test_pipeline_passes_schema_and_question(mock_gen, schema, df) -> None:
    mock_gen.return_value = _generated("result = 1")
    run_pipeline(schema, "how many rows?", df)

    mock_gen.assert_called_once_with(schema, "how many rows?")
