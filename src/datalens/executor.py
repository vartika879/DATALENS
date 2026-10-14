"""Execute LLM-generated pandas code against a DataFrame.

SECURITY NOTES
--------------
Two execution paths are supported:

1. **Direct exec** (``use_sandbox=False``, default): Uses Python's built-in
   ``exec`` with a restricted ``__builtins__`` dict. This is NOT a security
   boundary. It stops accidental ``import``/``open`` calls but can be
   escaped via object introspection.

2. **Docker sandbox** (``use_sandbox=True``): Dispatches to
   ``datalens.sandbox.run_in_sandbox`` for real isolation (no network,
   resource limits, read-only filesystem, non-root user).

Both paths run an AST static check first (``datalens.ast_check``). The AST
check is a cheap first layer — it is NOT a security boundary either, but it
rejects obvious attacks before spawning a container or calling ``exec``.
"""

from __future__ import annotations


from datalens.logging_config import get_logger

log = get_logger("datalens.executor")

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from datalens.ast_check import ASTCheckError, assert_code_is_safe
from datalens.sandbox import SandboxResult, run_in_sandbox
from datalens.serialization import sanitize_for_json

# --- Limits -----------------------------------------------------------------

MAX_RESULT_ROWS = 100
MAX_RESULT_REPR_CHARS = 10_000

# --- Restricted builtins (direct-exec path only) ----------------------------

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

    status: Literal["success", "error", "timeout"]
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
    """Return (result_type, result_data) for an arbitrary result value.

    Handles:
    - DataFrame / Series → structured dict
    - None → "none"
    - list / tuple / numpy array → "list" with sanitized elements
    - pandas extension arrays (ArrowStringArray, Categorical) → "list"
    - scalars → "scalar"
    """
    if isinstance(value, pd.DataFrame):
        return "dataframe", _serialize_dataframe(value)
    if isinstance(value, pd.Series):
        return "series", _serialize_series(value)
    if value is None:
        return "none", None

    # Python sequences and numpy arrays → list
    if isinstance(value, (list, tuple, np.ndarray)):
        return "list", [sanitize_for_json(v) for v in value]

    # pandas extension arrays (ArrowStringArray, Categorical, etc.) expose
    # .tolist() and .dtype but are not Series. Convert to list.
    if hasattr(value, "tolist") and hasattr(value, "dtype"):
        try:
            return "list", [sanitize_for_json(v) for v in value.tolist()]
        except Exception:  # noqa: BLE001 — fall through to scalar
            pass

    return "scalar", sanitize_for_json(value)


def _format_repr(value: Any) -> str:
    """Short, human-readable representation, capped at MAX_RESULT_REPR_CHARS.

    numpy scalars are converted to native Python first so the repr is clean
    (``57500.0`` instead of ``np.float64(57500.0)``). DataFrames and Series
    keep their natural pandas repr.
    """
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = sanitize_for_json(value)
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 — repr can raise on odd objects
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > MAX_RESULT_REPR_CHARS:
        text = text[:MAX_RESULT_REPR_CHARS] + "... (truncated)"
    return text


def _sandbox_result_to_execution(sr: SandboxResult) -> ExecutionResult:
    """Adapt a SandboxResult to the common ExecutionResult interface."""
    return ExecutionResult(
        status=sr.status,
        result_type=sr.result_type,
        result_repr=sr.result_repr,
        result_data=sr.result_data,
        error_type=sr.error_type,
        error_message=sr.error_message,
        error_traceback=sr.error_traceback,
        duration_ms=sr.duration_ms,
    )


# --- Public API -------------------------------------------------------------


def execute_code(
    code: str,
    df: pd.DataFrame,
    *,
    use_sandbox: bool = False,
    csv_path: Path | None = None,
) -> ExecutionResult:
    """Run ``code`` against ``df`` and capture the ``result`` variable.

    Args:
        code: Python source to execute.
        df: DataFrame exposed to the code as ``df`` (used in direct-exec mode).
        use_sandbox: If True, dispatch to the Docker sandbox instead of exec.
            Requires ``csv_path`` to be provided.
        csv_path: Host path to the CSV (required for sandbox mode).

    Returns:
        ExecutionResult.
    """
    log.info(
        "executor.started",
        use_sandbox=use_sandbox,
        code_length=len(code),
    )
    # --- Sandbox path (real isolation) --------------------------------------
    if use_sandbox:
        if csv_path is None:
            return ExecutionResult(
                status="error",
                error_type="SandboxConfigError",
                error_message=(
                    "use_sandbox=True requires csv_path. "
                    "Register the dataset with its source CSV path."
                ),
            )
        # AST check before spawning a container (cheap fast-fail).
        try:
            assert_code_is_safe(code)
        except SyntaxError as exc:
            return ExecutionResult(
                status="error",
                error_type="SyntaxError",
                error_message=str(exc),
            )
        except ASTCheckError as exc:
            return ExecutionResult(
                status="error",
                error_type="ASTCheckError",
                error_message=str(exc),
            )

        sandbox_result = run_in_sandbox(code, csv_path)
        return _sandbox_result_to_execution(sandbox_result)

    # --- Direct exec path (dev/test only) -----------------------------------
    started = time.perf_counter()

    # AST check before compiling.
    try:
        assert_code_is_safe(code)
    except SyntaxError as exc:
        return ExecutionResult(
            status="error",
            error_type="SyntaxError",
            error_message=str(exc),
            error_traceback=traceback.format_exc(),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
    except ASTCheckError as exc:
        return ExecutionResult(
            status="error",
            error_type="ASTCheckError",
            error_message=str(exc),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

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
        exec(compiled, sandbox)  # noqa: S102 — sandbox path has real isolation
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
    log.info(
        "executor.completed",
        result_type=result_type,
        duration_ms=round(result.duration_ms, 1),
    )
