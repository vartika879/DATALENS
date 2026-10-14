"""End-to-end pipeline: schema → generate code → execute.

This module is the single place where the three stages are wired together.
The Streamlit app (and later the FastAPI backend) call `run_pipeline`
without knowing the individual pieces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from datalens.agent.graph import build_agent_graph
from datalens.executor import ExecutionResult, execute_code
from datalens.llm import GeneratedCode, LLMError, generate_code
from datalens.logging_config import get_logger
from datalens.schema import DatasetSchema
from datalens.session import append_turn

log = get_logger("datalens.pipeline")


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
    log.info("pipeline.started", question_length=len(question))
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

    status = "error" if execution.status == "error" else "success"
    log.info(
        "pipeline.completed",
        status=status,
        total_ms=round(total_ms, 1),
    )

    if execution.status == "error":
        total_ms = (time.perf_counter() - started) * 1000.0
        log.info(
            "pipeline.completed",
            status=status,
            total_ms=round(total_ms, 1),
        )
        return PipelineResult(
            status="error",
            generated=generated,
            execution=execution,
            error_message=execution.error_message,
            total_duration_ms=total_ms,
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        log.info(
            "pipeline.completed",
            status=status,
            total_ms=round(total_ms, 1),
        )

    return PipelineResult(
        status="success",
        generated=generated,
        execution=execution,
        total_duration_ms=total_ms,
    )


def run_conversation(
    session_id: str,
    dataset_id: str,
    question: str,
) -> dict:
    """Run one turn of the agentic conversation and persist the turn.
    ...
    """
    started = time.perf_counter()

    app = build_agent_graph()
    result = app.invoke(
        {
            "session_id": session_id,
            "dataset_id": dataset_id,
            "question": question,
        }
    )

    # Attach wall-clock timing to the result so callers (eval runner,
    # API, UI) can report honest latency.
    result["total_duration_ms"] = (time.perf_counter() - started) * 1000.0

    final = result.get("final_answer")
    if final:
        append_turn(session_id, question, final)

    return result
