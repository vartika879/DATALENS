"""Shared state for the DataLens agent graph.

The state flows through every node in the workflow. Nodes read from and
write to these fields. Field names use a plain vocabulary so that node
implementations stay easy to reason about.

IMPORTANT: never place a pandas DataFrame in this state. DataFrames do not
serialize cleanly through LangGraph, and holding them in state causes
memory bloat. Only a `dataset_id` reference belongs here.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between nodes in the DataLens agent.

    All fields are optional (``total=False``) so that partial updates from
    individual nodes are valid LangGraph returns.
    """

    # --- Input ----------------------------------------------------------
    dataset_id: str
    session_id: str
    question: str
    # --- Planner output -------------------------------------------------
    resolved_question: str
    planner_decision: str
    planner_reasoning: str
    plan: str
    clarification_needed: str
    conversation_history: list[dict]

    # --- Coder output ---------------------------------------------------
    code: str
    explanation: str
    self_reported_confidence: float
    previous_code: str  # for fixer — last code attempted
    # --- Executor output ------------------------------------------------
    execution_status: str  # "success" | "error"
    result_type: str | None  # "scalar" | "dataframe" | "series" | "none"
    result_data: Any
    result_repr: str | None
    execution_error: str | None
    execution_traceback: str | None
    execution_duration_ms: float

    # --- Fixer ----------------------------------------------------------
    retry_count: int
    previous_errors: list[str]

    # --- Summarizer output ----------------------------------------------
    final_answer: str
