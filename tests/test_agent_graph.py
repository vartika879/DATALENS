"""Tests for the LangGraph agent skeleton."""

from __future__ import annotations

from datalens.agent.graph import build_agent_graph
from datalens.agent.nodes import (
    MAX_RETRIES,
    fixer_node,
    route_after_executor,
)
from datalens.agent.state import AgentState

# --- Graph construction -----------------------------------------------------


def test_graph_compiles() -> None:
    app = build_agent_graph()
    assert app is not None


# --- End-to-end happy path --------------------------------------------------


def test_graph_happy_path_produces_final_answer() -> None:
    app = build_agent_graph()
    result = app.invoke({"dataset_id": "ds1", "question": "any"})

    # Node outputs are merged into the state.
    assert result["is_answerable"] is True
    assert result["execution_status"] == "success"
    assert result["final_answer"].startswith("Result:")


# --- Routing logic ----------------------------------------------------------


def test_route_success_goes_to_summarizer() -> None:
    state: AgentState = {"execution_status": "success"}
    assert route_after_executor(state) == "summarizer"


def test_route_error_with_retries_left_goes_to_fixer() -> None:
    state: AgentState = {"execution_status": "error", "retry_count": 0}
    assert route_after_executor(state) == "fixer"


def test_route_error_at_max_retries_gives_up() -> None:
    state: AgentState = {
        "execution_status": "error",
        "retry_count": MAX_RETRIES,
    }
    assert route_after_executor(state) == "give_up"


# --- Fixer node -------------------------------------------------------------


def test_fixer_increments_retry_count_and_records_error() -> None:
    state: AgentState = {
        "retry_count": 0,
        "execution_error": "KeyError: 'nope'",
    }
    updates = fixer_node(state)
    assert updates["retry_count"] == 1
    assert updates["previous_errors"] == ["KeyError: 'nope'"]


def test_fixer_appends_to_existing_errors() -> None:
    state: AgentState = {
        "retry_count": 2,
        "execution_error": "second",
        "previous_errors": ["first"],
    }
    updates = fixer_node(state)
    assert updates["retry_count"] == 3
    assert updates["previous_errors"] == ["first", "second"]
