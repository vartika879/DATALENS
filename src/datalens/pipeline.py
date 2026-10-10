"""End-to-end pipeline: schema → generate code → execute.

This module is the single place where the three stages are wired together.
The Streamlit app (and later the FastAPI backend) call `run_pipeline`
without knowing the individual pieces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from datalens.executor import ExecutionResult, execute_code
from datalens.llm import GeneratedCode, LLMError, generate_code
from datalens.schema import DatasetSchema


@dataclass
class PipelineResult:
    """Full outcome of one question against one dataset."""

    status: str  # "success" | "error" | "llm_error"
    generated: GeneratedCode | None = None
    execution: ExecutionResult | None = None
    error_message: str | None = None
    total_duration_ms: float = 0.0


def run_pipeline(
    schema: DatasetSchema,
    question: str,
    df: pd.DataFrame,
) -> PipelineResult:
    """Run the full pipeline for one question."""
    started = time.perf_counter()

    try:
        generated = generate_code(schema, question)
    except LLMError as exc:
        return PipelineResult(
            status="llm_error",
            error_message=str(exc),
            total_duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    execution = execute_code(generated.code, df)
    total_ms = (time.perf_counter() - started) * 1000.0

    if execution.status == "error":
        return PipelineResult(
            status="error",
            generated=generated,
            execution=execution,
            error_message=execution.error_message,
            total_duration_ms=total_ms,
        )

    return PipelineResult(
        status="success",
        generated=generated,
        execution=execution,
        total_duration_ms=total_ms,
    )
