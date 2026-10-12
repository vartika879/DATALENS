"""Tests for the Docker sandbox wrapper.

These tests require Docker and a built sandbox image. They are skipped
automatically if either is missing, so the test suite still passes in
environments without Docker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datalens.sandbox import (
    SANDBOX_IMAGE,
    SandboxError,
    run_in_sandbox,
    sandbox_image_exists,
)

pytestmark = pytest.mark.skipif(
    not sandbox_image_exists(),
    reason=f"Sandbox image '{SANDBOX_IMAGE}' not built",
)


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(
        "city,sales\nMumbai,100\nDelhi,200\nBangalore,300\n",
        encoding="utf-8",
    )
    return path


# --- Success paths ----------------------------------------------------------


def test_scalar_result(csv_path: Path) -> None:
    result = run_in_sandbox("result = df['sales'].sum()", csv_path)
    assert result.status == "success"
    assert result.result_type == "scalar"
    assert result.result_data == 600


def test_dataframe_result(csv_path: Path) -> None:
    code = "result = df.groupby('city')['sales'].sum().reset_index()"
    result = run_in_sandbox(code, csv_path)
    assert result.status == "success"
    assert result.result_type == "dataframe"
    assert result.result_data["shape"] == [3, 2]


def test_series_result(csv_path: Path) -> None:
    result = run_in_sandbox("result = df['city'].value_counts()", csv_path)
    assert result.status == "success"
    assert result.result_type == "series"


# --- Error paths ------------------------------------------------------------


def test_syntax_error(csv_path: Path) -> None:
    result = run_in_sandbox("result = df[", csv_path)
    assert result.status == "error"
    assert result.error_type == "SyntaxError"


def test_runtime_error(csv_path: Path) -> None:
    result = run_in_sandbox("result = df['nope'].sum()", csv_path)
    assert result.status == "error"
    assert result.error_type == "KeyError"


def test_missing_result(csv_path: Path) -> None:
    result = run_in_sandbox("x = df['sales'].sum()", csv_path)
    assert result.status == "error"
    assert result.error_type == "MissingResultError"


# --- Security: restricted builtins block common escapes ---------------------


def test_import_is_blocked(csv_path: Path) -> None:
    result = run_in_sandbox("import os\nresult = os.getcwd()", csv_path)
    assert result.status == "error"
    # AST check inside the container blocks imports before exec.
    assert result.error_type == "ASTCheckError"


def test_open_is_blocked(csv_path: Path) -> None:
    result = run_in_sandbox("result = open('/etc/passwd').read()", csv_path)
    assert result.status == "error"
    assert result.error_type == "ASTCheckError"


def test_eval_is_blocked(csv_path: Path) -> None:
    result = run_in_sandbox("result = eval('1+1')", csv_path)
    assert result.status == "error"
    assert result.error_type == "ASTCheckError"


# --- Timeout ----------------------------------------------------------------


def test_timeout_kills_infinite_loop(csv_path: Path) -> None:
    result = run_in_sandbox("result = 1\nwhile True:\n    pass", csv_path, timeout=3)
    assert result.status == "timeout"


# --- Missing CSV ------------------------------------------------------------


def test_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="CSV not found"):
        run_in_sandbox("result = 1", tmp_path / "nope.csv")
