"""Node functions for the DataLens agent graph.

Each function takes the current AgentState and returns a partial dict of
updates. Phase 2.1 provides pass-through stubs so the graph can be tested
in isolation before real logic lands.
"""

from datalens.agent.state import AgentState
from datalens.config import get_settings
from datalens.executor import execute_code
from datalens.llm import LLMError, fix_code, generate_code, plan_question, summarize
from datalens.registry import get_dataset
from datalens.session import get_history

# Maximum number of repair attempts after the initial execution.
MAX_RETRIES = 3


def planner_node(state: AgentState) -> dict:
    """Resolve the question (using history) and decide how to proceed."""
    dataset = get_dataset(state["dataset_id"])

    session_id = state.get("session_id")
    history_turns = get_history(session_id) if session_id else []
    history_dicts = [{"question": t.question, "answer": t.final_answer} for t in history_turns]

    try:
        output = plan_question(dataset.schema, state["question"], history=history_dicts)
    except LLMError as exc:
        return {
            "planner_decision": "refuse",
            "planner_reasoning": f"planner failed: {exc}",
            "resolved_question": state["question"],
            "conversation_history": history_dicts,
            "final_answer": (
                "I couldn't plan an answer to this question. Please try rephrasing it."
            ),
        }

    updates: dict = {
        "resolved_question": output.resolved_question,
        "planner_decision": output.decision,
        "planner_reasoning": output.reasoning,
        "conversation_history": history_dicts,
    }

    if output.decision == "proceed":
        updates["plan"] = output.plan
    elif output.decision == "clarify":
        updates["clarification_needed"] = output.clarifying_question
        updates["final_answer"] = output.clarifying_question
    else:
        updates["final_answer"] = f"I can't answer this question: {output.reasoning}"

    return updates


def route_after_planner(state: AgentState) -> str:
    """Conditional edge: proceed to coder, or exit early."""
    if state.get("planner_decision") == "proceed":
        return "coder"
    return "end"


def coder_node(state: AgentState) -> dict:
    """Generate pandas code for the question using the dataset schema."""
    dataset = get_dataset(state["dataset_id"])

    try:
        question = state.get("resolved_question") or state["question"]
        generated = generate_code(dataset.schema, question)
    except LLMError as exc:
        return {
            "code": "",
            "explanation": "",
            "self_reported_confidence": 0.0,
            "previous_code": state.get("previous_code", ""),
            "execution_status": "error",
            "execution_error": f"coder failed: {exc}",
            "final_answer": (
                "I couldn't generate code for this question. Please try rephrasing it."
            ),
        }

    return {
        "code": generated.code,
        "explanation": generated.explanation,
        "self_reported_confidence": generated.self_reported_confidence,
        "previous_code": generated.code,
    }


def executor_node(state: AgentState) -> dict:
    """Execute the generated code against the dataset.

    Handles three cases:
    1. Code is empty (coder failed) — propagate the error.
    2. execute_code returns success — store the serialized result.
    3. execute_code returns error — store the traceback for the fixer.
    """
    code = state.get("code") or ""

    # Case 1: coder failed and left no code.
    if not code.strip():
        return {
            "execution_status": "error",
            "execution_error": state.get("execution_error") or "no code to execute",
            "execution_traceback": None,
            "execution_duration_ms": 0.0,
        }

    # Case 2 & 3: run the code.
    dataset = get_dataset(state["dataset_id"])
    settings = get_settings()
    result = execute_code(
        code,
        dataset.df,
        use_sandbox=settings.use_sandbox,
        csv_path=dataset.csv_path,
    )
    if result.status == "success":
        return {
            "execution_status": "success",
            "result_type": result.result_type,
            "result_data": result.result_data,
            "result_repr": result.result_repr,
            "execution_error": None,
            "execution_traceback": None,
            "execution_duration_ms": result.duration_ms,
        }

    return {
        "execution_status": "error",
        "result_type": None,
        "result_data": None,
        "result_repr": None,
        "execution_error": result.error_message,
        "execution_traceback": result.error_traceback,
        "execution_duration_ms": result.duration_ms,
    }


def fixer_node(state: AgentState) -> dict:
    """Attempt to repair code that failed execution.

    Reads the failed code, its traceback, and any earlier errors, then asks
    the LLM for a corrected version. Increments retry_count and records the
    error so the next attempt can see what has already been tried.
    """
    dataset = get_dataset(state["dataset_id"])

    failed_code = state.get("code", "")
    error_message = state.get("execution_error") or "unknown error"
    error_traceback = state.get("execution_traceback")
    previous_errors = list(state.get("previous_errors", []))

    try:
        fixed = fix_code(
            schema=dataset.schema,
            question=state["question"],
            previous_code=failed_code,
            error_message=error_message,
            error_traceback=error_traceback,
            previous_errors=previous_errors,
        )
    except LLMError as exc:
        # Fixer itself failed — record and let the router decide.
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "previous_errors": previous_errors + [error_message],
            "execution_error": f"fixer failed: {exc}",
        }

    return {
        "code": fixed.code,
        "explanation": fixed.explanation,
        "previous_code": failed_code,
        "retry_count": state.get("retry_count", 0) + 1,
        "previous_errors": previous_errors + [error_message],
        # Reset executor outputs so the retry starts clean.
        "execution_status": "",
        "result_type": None,
        "result_data": None,
        "result_repr": None,
        "execution_error": None,
        "execution_traceback": None,
    }


def summarizer_node(state: AgentState) -> dict:
    """Produce a natural-language answer for the user.

    If the planner already set a final_answer (clarify/refuse cases),
    respect it — the summarizer only runs on the success path.

    If the LLM summarizer fails, fall back to a plain ``Result: X``
    so the user still sees something useful.
    """
    # Respect a planner-provided final answer if present.
    if state.get("final_answer"):
        return {}

    dataset = get_dataset(state["dataset_id"])

    result_repr = state.get("result_repr") or ""
    explanation = state.get("explanation") or ""

    try:
        summary = summarize(
            schema=dataset.schema,
            question=state["question"],
            result_repr=result_repr,
            explanation=explanation,
        )
    except LLMError:
        return {"final_answer": f"Result: {result_repr}"}

    return {"final_answer": summary.answer}


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
