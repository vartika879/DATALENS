"""Sandbox runner — executes pandas code inside the isolated container.

Reads a JSON payload from stdin, executes the code against the CSV at
/data/input.csv, and prints a JSON result to stdout.

SECURITY NOTE: This script runs INSIDE the sandbox container. The
container itself provides the isolation (no network, no root, read-only
filesystem, resource limits). This script does not need to be bulletproof
— its job is to run code, and the container constrains what the code can do.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

MAX_RESULT_ROWS = 100
MAX_RESULT_REPR_CHARS = 10_000
CSV_PATH = "/data/input.csv"


def _sanitize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _serialize_dataframe(df: pd.DataFrame) -> dict:
    truncated = len(df) > MAX_RESULT_ROWS
    head = df.head(MAX_RESULT_ROWS)
    records = [
        {str(k): _sanitize(v) for k, v in row.items()} for row in head.to_dict(orient="records")
    ]
    return {
        "columns": [str(c) for c in df.columns],
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "rows": records,
        "truncated": truncated,
    }


def _serialize_series(series: pd.Series) -> dict:
    truncated = len(series) > MAX_RESULT_ROWS
    head = series.head(MAX_RESULT_ROWS)
    return {
        "name": None if series.name is None else str(series.name),
        "values": [_sanitize(v) for v in head.tolist()],
        "truncated": truncated,
    }


def _serialize_result(value: Any) -> tuple[str, Any]:
    if isinstance(value, pd.DataFrame):
        return "dataframe", _serialize_dataframe(value)
    if isinstance(value, pd.Series):
        return "series", _serialize_series(value)
    if value is None:
        return "none", None
    return "scalar", _sanitize(value)


def _format_repr(value: Any) -> str:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = _sanitize(value)
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > MAX_RESULT_REPR_CHARS:
        text = text[:MAX_RESULT_REPR_CHARS] + "... (truncated)"
    return text


def _error_response(exc: BaseException, started: float) -> dict:
    return {
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_traceback": traceback.format_exc(),
        "duration_ms": (time.perf_counter() - started) * 1000.0,
    }


def main() -> None:
    started = time.perf_counter()

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
        code = payload["code"]
    except Exception as exc:
        print(json.dumps(_error_response(exc, started)))
        return

    # Allowed builtins — second layer of defense (container is the first).
    allowed_builtins: dict[str, Any] = {
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

    # Load the CSV inside the container.
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as exc:
        print(json.dumps(_error_response(exc, started)))
        return

    sandbox: dict[str, Any] = {
        "__builtins__": allowed_builtins,
        "pd": pd,
        "np": np,
        "df": df,
    }

    try:
        compiled = compile(code, "<sandbox>", "exec")
    except SyntaxError as exc:
        print(json.dumps(_error_response(exc, started)))
        return

    try:
        exec(compiled, sandbox)  # noqa: S102 — container provides isolation
    except Exception as exc:
        print(json.dumps(_error_response(exc, started)))
        return

    if "result" not in sandbox:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "MissingResultError",
                    "error_message": "Code did not assign a variable named 'result'",
                    "error_traceback": None,
                    "duration_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
        )
        return

    raw_result = sandbox["result"]
    result_type, result_data = _serialize_result(raw_result)

    print(
        json.dumps(
            {
                "status": "success",
                "result_type": result_type,
                "result_data": result_data,
                "result_repr": _format_repr(raw_result),
                "duration_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    )


if __name__ == "__main__":
    main()
