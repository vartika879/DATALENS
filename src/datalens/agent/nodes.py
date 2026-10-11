"""Node functions for the DataLens agent graph.

Each function takes the current AgentState and returns a partial dict of
updates. Phase 2.1 provides pass-through stubs so the graph can be tested
in isolation before real logic lands.
"""

from __future__ import annotations

from datalens.agent.state import AgentState

# Maximum number of repair attempts after the initial execution.
MAX_RETRIES = 3


def planner_node(state: AgentState) -> dict:
    """Decide whether the question is answerable and outline a plan."""
    # Stub: assume answerable.
    return {"is_answerable": True, "plan": "stub plan"}


def coder_node(state: AgentState) -> dict:
    """Generate pandas code for the question."""
    # Stub: emit a trivial result variable.
    return {
        "code": "result = 0",
        "explanation": "stub explanation",
        "self_reported_confidence": 0.5,
    }


def executor_node(state: AgentState) -> dict:
    """Execute the generated code."""
    # Stub: pretend success.
    return {
        "execution_status": "success",
        "result_type": "scalar",
        "result_data": 0,
        "result_repr": "0",
        "execution_duration_ms": 0.0,
    }


def fixer_node(state: AgentState) -> dict:
    """Attempt to repair a failed execution."""
    retries = state.get("retry_count", 0) + 1
    errors = list(state.get("previous_errors", []))
    errors.append(state.get("execution_error") or "unknown error")
    return {"retry_count": retries, "previous_errors": errors}


def summarizer_node(state: AgentState) -> dict:
    """Produce a final, user-facing answer."""
    # Stub: echo the result.
    return {"final_answer": f"Result: {state.get('result_repr')}"}


def route_after_executor(state: AgentState) -> str:
    """Conditional edge: decide where to go after executor runs.

    Returns one of: "summarizer", "fixer", "give_up".
    """
    if state.get("execution_status") == "success":
        return "summarizer"

    retries = state.get("retry_count", 0)
    if retries < MAX_RETRIES:
        return "fixer"

    return "give_up"
