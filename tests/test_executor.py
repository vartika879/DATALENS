"""Tests for the code executor.

These tests exercise the restricted-builtins layer and result serialization.
They do NOT prove the executor is secure — see the module docstring.
"""

from __future__ import annotations

import pandas as pd
import pytest

from datalens.executor import MAX_RESULT_ROWS, execute_code


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["Mumbai", "Delhi", "Mumbai", "Bangalore", "Delhi"],
            "sales": [100, 200, 150, 300, 250],
        }
    )


# --- Success cases ----------------------------------------------------------


def test_scalar_result(df: pd.DataFrame) -> None:
    result = execute_code("result = df['sales'].sum()", df)
    assert result.status == "success"
    assert result.result_type == "scalar"
    assert result.result_data == 1000
    assert result.error_type is None
    assert result.duration_ms >= 0.0


def test_dataframe_result(df: pd.DataFrame) -> None:
    code = "result = df.groupby('city')['sales'].sum().reset_index()"
    result = execute_code(code, df)

    assert result.status == "success"
    assert result.result_type == "dataframe"
    assert set(result.result_data["columns"]) == {"city", "sales"}
    assert result.result_data["shape"] == [3, 2]
    assert result.result_data["truncated"] is False


def test_dataframe_result_truncated_when_too_large(df: pd.DataFrame) -> None:
    big = pd.concat([df] * 100, ignore_index=True)  # 500 rows
    result = execute_code("result = df", big)

    assert result.status == "success"
    assert result.result_type == "dataframe"
    assert result.result_data["truncated"] is True
    assert len(result.result_data["rows"]) == MAX_RESULT_ROWS


def test_series_result(df: pd.DataFrame) -> None:
    result = execute_code("result = df['city'].value_counts()", df)

    assert result.status == "success"
    assert result.result_type == "series"
    assert result.result_data["name"] == "count"
    assert len(result.result_data["values"]) == 3


def test_none_result(df: pd.DataFrame) -> None:
    result = execute_code("result = None", df)

    assert result.status == "success"
    assert result.result_type == "none"
    assert result.result_data is None


# --- Error cases ------------------------------------------------------------


def test_syntax_error_is_captured(df: pd.DataFrame) -> None:
    result = execute_code("result = df[", df)

    assert result.status == "error"
    assert result.error_type == "SyntaxError"
    assert result.error_traceback is not None


def test_runtime_error_is_captured(df: pd.DataFrame) -> None:
    result = execute_code("result = df['nope'].sum()", df)

    assert result.status == "error"
    assert result.error_type == "KeyError"
    assert "nope" in (result.error_message or "")
    assert result.error_traceback is not None


def test_missing_result_variable_is_captured(df: pd.DataFrame) -> None:
    result = execute_code("x = df['sales'].sum()", df)

    assert result.status == "error"
    assert result.error_type == "MissingResultError"
    assert "result" in (result.error_message or "")


# --- Restricted builtins ----------------------------------------------------


def test_import_is_blocked(df: pd.DataFrame) -> None:
    result = execute_code("import os\nresult = os.getcwd()", df)

    assert result.status == "error"
    assert result.error_type in {"ImportError", "ModuleNotFoundError"}


def test_open_is_blocked(df: pd.DataFrame) -> None:
    result = execute_code("result = open('/etc/passwd').read()", df)

    assert result.status == "error"
    assert result.error_type == "NameError"


def test_eval_is_blocked(df: pd.DataFrame) -> None:
    result = execute_code("result = eval('1+1')", df)

    assert result.status == "error"
    assert result.error_type == "NameError"


# --- Allowed builtins -------------------------------------------------------


def test_len_is_allowed(df: pd.DataFrame) -> None:
    result = execute_code("result = len(df)", df)
    assert result.status == "success"
    assert result.result_data == 5


def test_sum_is_allowed(df: pd.DataFrame) -> None:
    result = execute_code("result = sum([1, 2, 3])", df)
    assert result.status == "success"
    assert result.result_data == 6


def test_result_repr_is_present_on_success(df: pd.DataFrame) -> None:
    result = execute_code("result = 42", df)
    assert result.result_repr == "42"


def test_result_repr_is_present_on_success(df: pd.DataFrame) -> None:
    result = execute_code("result = 42", df)
    assert result.result_repr == "42"


def test_numpy_scalar_repr_is_clean(df: pd.DataFrame) -> None:
    """numpy scalars must not leak into repr — should be plain Python numbers."""
    result = execute_code("result = df['sales'].sum()", df)
    assert "np." not in (result.result_repr or "")
    assert "numpy" not in (result.result_repr or "")