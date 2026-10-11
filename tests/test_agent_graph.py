"""Tests for the LangGraph agent."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from datalens.agent.graph import build_agent_graph
from datalens.agent.nodes import (
    MAX_RETRIES,
    coder_node,
    executor_node,
    fixer_node,
    route_after_executor,
    route_after_planner,
    summarizer_node,
)
from datalens.agent.state import AgentState
from datalens.llm import (
    FixedCode,
    GeneratedCode,
    LLMError,
    PlannerOutput,
    SummaryOutput,
)
from datalens.registry import clear_registry, register_dataset
from datalens.schema import ColumnInfo, DatasetSchema


@pytest.fixture(autouse=True)
def _clean() -> None:
    clear_registry()


@pytest.fixture
def dataset_id() -> str:
    df = pd.DataFrame({"sales": [10, 20, 30]})
    schema = DatasetSchema(
        file_name="t.csv",
        row_count=3,
        column_count=1,
        columns=[ColumnInfo(name="sales", dtype="int64", null_count=0, null_percentage=0.0)],
        sample_rows=[{"sales": 10}],
    )
    return register_dataset(df, schema)


def _planner(decision: str, **kw) -> PlannerOutput:
    defaults = {
        "decision": decision,
        "plan": "plan" if decision == "proceed" else "",
        "clarifying_question": "which metric?" if decision == "clarify" else "",
        "reasoning": "r",
    }
    defaults.update(kw)
    return PlannerOutput(**defaults)


# --- Graph construction -----------------------------------------------------


def test_graph_compiles() -> None:
    assert build_agent_graph() is not None


# --- Happy path -------------------------------------------------------------


@patch("datalens.agent.nodes.summarize")
@patch("datalens.agent.nodes.generate_code")
@patch("datalens.agent.nodes.plan_question")
def test_graph_proceed_path(mock_plan, mock_gen, mock_sum, dataset_id: str) -> None:
    mock_plan.return_value = _planner("proceed")
    mock_gen.return_value = GeneratedCode(
        code="result = df['sales'].sum()",
        explanation="sum of sales",
        self_reported_confidence=0.9,
    )
    mock_sum.return_value = SummaryOutput(answer="Total sales are 60.")
    app = build_agent_graph()
    result = app.invoke({"dataset_id": dataset_id, "question": "any"})

    assert result["planner_decision"] == "proceed"
    assert result["code"] == "result = df['sales'].sum()"
    assert result["execution_status"] == "success"
    assert result["result_data"] == 60
    assert result["final_answer"] == "Total sales are 60."


@patch("datalens.agent.nodes.plan_question")
def test_graph_clarify_path(mock_plan, dataset_id: str) -> None:
    mock_plan.return_value = _planner("clarify")
    app = build_agent_graph()
    result = app.invoke({"dataset_id": dataset_id, "question": "best?"})

    assert result["planner_decision"] == "clarify"
    assert result["final_answer"] == "which metric?"
    assert "code" not in result  # coder never ran


@patch("datalens.agent.nodes.plan_question")
def test_graph_refuse_path(mock_plan, dataset_id: str) -> None:
    mock_plan.return_value = _planner("refuse", reasoning="no profit column")
    app = build_agent_graph()
    result = app.invoke({"dataset_id": dataset_id, "question": "profit?"})

    assert result["planner_decision"] == "refuse"
    assert "no profit column" in result["final_answer"]
    assert "code" not in result


# --- Routing logic ----------------------------------------------------------


def test_route_planner_proceed() -> None:
    assert route_after_planner({"planner_decision": "proceed"}) == "coder"


def test_route_planner_clarify() -> None:
    assert route_after_planner({"planner_decision": "clarify"}) == "end"


def test_route_planner_refuse() -> None:
    assert route_after_planner({"planner_decision": "refuse"}) == "end"


def test_route_success_goes_to_summarizer() -> None:
    assert route_after_executor({"execution_status": "success"}) == "summarizer"


def test_route_error_with_retries_left_goes_to_fixer() -> None:
    state: AgentState = {"execution_status": "error", "retry_count": 0}
    assert route_after_executor(state) == "fixer"


def test_route_error_at_max_retries_gives_up() -> None:
    state: AgentState = {"execution_status": "error", "retry_count": MAX_RETRIES}
    assert route_after_executor(state) == "give_up"


# --- Fixer ------------------------------------------------------------------


# --- Coder node -------------------------------------------------------------


@patch("datalens.agent.nodes.generate_code")
def test_coder_node_success(mock_gen, dataset_id: str) -> None:
    mock_gen.return_value = GeneratedCode(
        code="result = df['sales'].mean()",
        explanation="mean of sales",
        self_reported_confidence=0.9,
    )
    updates = coder_node({"dataset_id": dataset_id, "question": "any"})

    assert updates["code"] == "result = df['sales'].mean()"
    assert updates["explanation"] == "mean of sales"
    assert updates["self_reported_confidence"] == pytest.approx(0.9)
    assert updates["previous_code"] == "result = df['sales'].mean()"


@patch("datalens.agent.nodes.generate_code")
def test_coder_node_llm_error(mock_gen, dataset_id: str) -> None:
    mock_gen.side_effect = LLMError("api down")
    updates = coder_node({"dataset_id": dataset_id, "question": "any"})

    assert updates["code"] == ""
    assert updates["execution_status"] == "error"
    assert "coder failed" in updates["execution_error"]
    assert "final_answer" in updates


@patch("datalens.agent.nodes.generate_code")
def test_coder_node_preserves_previous_code_on_error(mock_gen, dataset_id: str) -> None:
    mock_gen.side_effect = LLMError("boom")
    updates = coder_node(
        {
            "dataset_id": dataset_id,
            "question": "any",
            "previous_code": "old code",
        }
    )
    assert updates["previous_code"] == "old code"


# --- Executor node ----------------------------------------------------------


def test_executor_node_success(dataset_id: str) -> None:
    updates = executor_node({"dataset_id": dataset_id, "code": "result = df['sales'].sum()"})
    assert updates["execution_status"] == "success"
    assert updates["result_data"] == 60
    assert updates["result_type"] == "scalar"
    assert updates["execution_error"] is None
    assert updates["execution_duration_ms"] >= 0.0


def test_executor_node_captures_runtime_error(dataset_id: str) -> None:
    updates = executor_node({"dataset_id": dataset_id, "code": "result = df['nope'].sum()"})
    assert updates["execution_status"] == "error"
    assert updates["result_data"] is None
    assert "nope" in (updates["execution_error"] or "")
    assert updates["execution_traceback"] is not None


def test_executor_node_empty_code_returns_error(dataset_id: str) -> None:
    updates = executor_node({"dataset_id": dataset_id, "code": ""})
    assert updates["execution_status"] == "error"
    assert "no code to execute" in (updates["execution_error"] or "")


def test_executor_node_preserves_coder_error(dataset_id: str) -> None:
    updates = executor_node(
        {
            "dataset_id": dataset_id,
            "code": "",
            "execution_error": "coder failed: api down",
        }
    )
    assert updates["execution_status"] == "error"
    assert updates["execution_error"] == "coder failed: api down"


def test_executor_node_missing_result_variable(dataset_id: str) -> None:
    updates = executor_node({"dataset_id": dataset_id, "code": "x = df['sales'].sum()"})
    assert updates["execution_status"] == "error"
    assert "result" in (updates["execution_error"] or "")


# --- Fixer node (real) ------------------------------------------------------


@patch("datalens.agent.nodes.fix_code")
def test_fixer_node_updates_code_and_increments_retry(mock_fix, dataset_id: str) -> None:
    mock_fix.return_value = FixedCode(
        code="result = df['sales'].sum()",
        explanation="fixed sum",
    )
    updates = fixer_node(
        {
            "dataset_id": dataset_id,
            "question": "any",
            "code": "result = df['sales'].summ()",
            "execution_error": "AttributeError: summ",
            "execution_traceback": "tb",
        }
    )

    assert updates["code"] == "result = df['sales'].sum()"
    assert updates["previous_code"] == "result = df['sales'].summ()"
    assert updates["retry_count"] == 1
    assert updates["previous_errors"] == ["AttributeError: summ"]
    assert updates["execution_status"] == ""


@patch("datalens.agent.nodes.fix_code")
def test_fixer_node_accumulates_previous_errors(mock_fix, dataset_id: str) -> None:
    mock_fix.return_value = FixedCode(code="new", explanation="e")
    updates = fixer_node(
        {
            "dataset_id": dataset_id,
            "question": "any",
            "code": "bad",
            "execution_error": "second",
            "previous_errors": ["first"],
        }
    )
    assert updates["previous_errors"] == ["first", "second"]
    assert updates["retry_count"] == 1


@patch("datalens.agent.nodes.fix_code")
def test_fixer_node_handles_llm_error(mock_fix, dataset_id: str) -> None:
    mock_fix.side_effect = LLMError("api down")
    updates = fixer_node(
        {
            "dataset_id": dataset_id,
            "question": "any",
            "code": "bad",
            "execution_error": "orig error",
        }
    )
    assert updates["retry_count"] == 1
    assert "fixer failed" in updates["execution_error"]


# --- End-to-end repair loop -------------------------------------------------
@patch("datalens.agent.nodes.summarize")
@patch("datalens.agent.nodes.fix_code")
@patch("datalens.agent.nodes.generate_code")
@patch("datalens.agent.nodes.plan_question")
def test_graph_repairs_failing_code(
    mock_plan, mock_gen, mock_fix, mock_sum, dataset_id: str
) -> None:
    """First coder output fails; fixer produces valid code; graph succeeds."""
    mock_plan.return_value = _planner("proceed")
    mock_gen.return_value = GeneratedCode(
        code="result = df['nonexistent'].sum()",
        explanation="bad",
        self_reported_confidence=0.5,
    )
    mock_fix.return_value = FixedCode(
        code="result = df['sales'].sum()",
        explanation="fixed",
    )
    mock_sum.return_value = SummaryOutput(answer="Total sales are 60.")

    app = build_agent_graph()
    result = app.invoke({"dataset_id": dataset_id, "question": "any"})

    assert result["execution_status"] == "success"
    assert result["result_data"] == 60
    assert result["retry_count"] == 1
    assert len(result["previous_errors"]) == 1


# --- Summarizer node (real) -------------------------------------------------


@patch("datalens.agent.nodes.summarize")
def test_summarizer_node_returns_llm_answer(mock_sum, dataset_id: str) -> None:
    mock_sum.return_value = SummaryOutput(answer="The total sales are 60.")
    updates = summarizer_node(
        {
            "dataset_id": dataset_id,
            "question": "total sales?",
            "result_repr": "60",
            "explanation": "sum of sales",
        }
    )
    assert updates["final_answer"] == "The total sales are 60."


@patch("datalens.agent.nodes.summarize")
def test_summarizer_node_falls_back_on_llm_error(mock_sum, dataset_id: str) -> None:
    mock_sum.side_effect = LLMError("api down")
    updates = summarizer_node(
        {
            "dataset_id": dataset_id,
            "question": "q",
            "result_repr": "42",
            "explanation": "e",
        }
    )
    assert updates["final_answer"] == "Result: 42"


def test_summarizer_node_respects_existing_final_answer(
    dataset_id: str,
) -> None:
    updates = summarizer_node(
        {
            "dataset_id": dataset_id,
            "question": "q",
            "final_answer": "already set",
        }
    )
    assert updates == {}
