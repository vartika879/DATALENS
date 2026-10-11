"""Tests for the LangGraph agent."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from datalens.agent.graph import build_agent_graph
from datalens.agent.nodes import (
    MAX_RETRIES,
    fixer_node,
    route_after_executor,
    route_after_planner,
)
from datalens.agent.state import AgentState
from datalens.llm import GeneratedCode, LLMError, PlannerOutput
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


@patch("datalens.agent.nodes.generate_code")
@patch("datalens.agent.nodes.plan_question")
def test_graph_proceed_path(mock_plan, mock_gen, dataset_id: str) -> None:
    mock_plan.return_value = _planner("proceed")
    mock_gen.return_value = GeneratedCode(
        code="result = df['sales'].sum()",
        explanation="sum of sales",
        self_reported_confidence=0.9,
    )
    app = build_agent_graph()
    result = app.invoke({"dataset_id": dataset_id, "question": "any"})

    assert result["planner_decision"] == "proceed"
    assert result["code"] == "result = df['sales'].sum()"
    assert result["execution_status"] == "success"
    assert result["final_answer"].startswith("Result:")


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


def test_fixer_increments_retry_count_and_records_error() -> None:
    updates = fixer_node({"retry_count": 0, "execution_error": "KeyError"})
    assert updates["retry_count"] == 1
    assert updates["previous_errors"] == ["KeyError"]


def test_fixer_appends_to_existing_errors() -> None:
    updates = fixer_node(
        {"retry_count": 2, "execution_error": "second", "previous_errors": ["first"]}
    )
    assert updates["retry_count"] == 3
    assert updates["previous_errors"] == ["first", "second"]


# --- Coder node -------------------------------------------------------------

from datalens.agent.nodes import coder_node


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
