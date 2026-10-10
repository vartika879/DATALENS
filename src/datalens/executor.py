"""Execute LLM-generated pandas code against a DataFrame.

SECURITY WARNING
----------------
This module uses Python's built-in ``exec`` with a restricted ``__builtins__``
dict. This is NOT a security boundary. ``exec`` can be escaped via object
introspection (e.g. walking ``__class__.__mro__`` from any object). The
restricted builtins only stop accidental ``import`` / ``open`` calls; they
do not stop a determined attacker.

Real isolation is added in Phase 3 (Docker sandbox: no network, resource
limits, read-only filesystem, non-root user). Until then, do NOT run
untrusted code through this function in any environment that matters.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from datalens.serialization import sanitize_for_json

# --- Limits -----------------------------------------------------------------

MAX_RESULT_ROWS = 100
MAX_RESULT_REPR_CHARS = 10_000

# --- Restricted builtins ----------------------------------------------------

_ALLOWED_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


# --- Result model -----------------------------------------------------------


@dataclass
class ExecutionResult:
    """Outcome of running a snippet of pandas code.

    ``result_data`` is JSON-safe and size-limited. ``error_traceback`` is a
    full Python traceback string, suitable for the Phase 2 Fixer node.
    """

    status: Literal["success", "error"]
    result_type: str | None = None
    result_repr: str | None = None
    result_data: Any = None
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    duration_ms: float = 0.0


# --- Internal helpers -------------------------------------------------------


def _serialize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Convert a DataFrame to a size-limited, JSON-safe dict."""
    truncated = len(df) > MAX_RESULT_ROWS
    head = df.head(MAX_RESULT_ROWS)
    records = [
        {str(k): sanitize_for_json(v) for k, v in row.items()}
        for row in head.to_dict(orient="records")
    ]
    return {
        "columns": [str(c) for c in df.columns],
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "rows": records,
        "truncated": truncated,
    }


def _serialize_series(series: pd.Series) -> dict[str, Any]:
    """Convert a Series to a size-limited, JSON-safe dict."""
    truncated = len(series) > MAX_RESULT_ROWS
    head = series.head(MAX_RESULT_ROWS)
    return {
        "name": None if series.name is None else str(series.name),
        "values": [sanitize_for_json(v) for v in head.tolist()],
        "truncated": truncated,
    }


def _serialize_result(value: Any) -> tuple[str, Any]:
    """Return (result_type, result_data) for an arbitrary result value."""
    if isinstance(value, pd.DataFrame):
        return "dataframe", _serialize_dataframe(value)
    if isinstance(value, pd.Series):
        return "series", _serialize_series(value)
    if value is None:
        return "none", None
    return "scalar", sanitize_for_json(value)


def _format_repr(value: Any) -> str:
    """Short, human-readable representation, capped at MAX_RESULT_REPR_CHARS."""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 — repr can raise on odd objects
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > MAX_RESULT_REPR_CHARS:
        text = text[:MAX_RESULT_REPR_CHARS] + "... (truncated)"
    return text


# --- Public API -------------------------------------------------------------


def execute_code(code: str, df: pd.DataFrame) -> ExecutionResult:
    """Run ``code`` against ``df`` and capture the ``result`` variable.

    The code is expected to assign its final answer to a variable named
    ``result``. The DataFrame is available as ``df``. ``pd`` and ``np`` are
    importable by name (already imported).

    Args:
        code: Python source to execute.
        df: DataFrame exposed to the code as ``df``.

    Returns:
        ExecutionResult — either success with serialized data, or error with
        a full traceback string.
    """
    started = time.perf_counter()

    sandbox: dict[str, Any] = {
        "__builtins__": _ALLOWED_BUILTINS,
        "pd": pd,
        "np": np,
        "df": df,
    }

    try:
        compiled = compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        return ExecutionResult(
            status="error",
            error_type="SyntaxError",
            error_message=str(exc),
            error_traceback=traceback.format_exc(),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    try:
        exec(compiled, sandbox)  # noqa: S102 — real isolation lands in Phase 3
    except Exception as exc:  # noqa: BLE001 — any runtime failure is captured
        return ExecutionResult(
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_traceback=traceback.format_exc(),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    if "result" not in sandbox:
        return ExecutionResult(
            status="error",
            error_type="MissingResultError",
            error_message="Code did not assign a variable named 'result'",
            error_traceback=None,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    raw = sandbox["result"]
    result_type, result_data = _serialize_result(raw)

    return ExecutionResult(
        status="success",
        result_type=result_type,
        result_repr=_format_repr(raw),
        result_data=result_data,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )
