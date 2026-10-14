"""Groq LLM client wrapper.

Produces structured ``GeneratedCode`` objects. All Groq-specific code lives
here so the rest of the codebase depends only on ``generate_code``.
"""

from __future__ import annotations


from datalens.logging_config import get_logger

log = get_logger("datalens.llm")


from pathlib import Path
from typing import Literal

from groq import Groq
from pydantic import BaseModel, Field, ValidationError, model_validator

from datalens.config import get_settings
from datalens.schema import DatasetSchema

DEFAULT_MODEL = "openai/gpt-oss-20b"

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_FILE = "generate_code_v1.txt"
_FIXER_PROMPT_FILE = "fixer_v1.txt"
_SUMMARIZER_PROMPT_FILE = "summarizer_v1.txt"
_PLANNER_PROMPT_FILE = "planner_v2.txt"


class LLMError(Exception):
    """Raised when the LLM call fails or the response cannot be parsed."""


class GeneratedCode(BaseModel):
    """Structured output from the LLM.

    ``self_reported_confidence`` is the model's own estimate and is NOT a
    calibrated probability. Do not use it as ground truth.
    """

    code: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)
    self_reported_confidence: float = Field(..., ge=0.0, le=1.0)


PlannerDecision = Literal["proceed", "clarify", "refuse"]

_PLANNER_PROMPT_FILE = "planner_v2.txt"


class PlannerOutput(BaseModel):
    """Structured output from the planner LLM call."""

    resolved_question: str = Field(..., min_length=1)
    decision: PlannerDecision
    plan: str = ""
    clarifying_question: str = ""
    reasoning: str = ""

    @model_validator(mode="after")
    def _check_fields_for_decision(self) -> PlannerOutput:
        if self.decision == "proceed" and not self.plan.strip():
            raise ValueError("plan is required when decision is 'proceed'")
        if self.decision == "clarify" and not self.clarifying_question.strip():
            raise ValueError("clarifying_question is required when decision is 'clarify'")
        return self


def plan_question(
    schema: DatasetSchema,
    question: str,
    history: list[dict] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    client: Groq | None = None,
) -> PlannerOutput:
    """Ask the LLM how to handle a question against a schema.

    Args:
        schema: The dataset schema.
        question: Natural-language question.
        model: Groq model name.
        client: Optional pre-built Groq client (for tests).

    Returns:
        A validated PlannerOutput.

    Raises:
        LLMError: On API failure, empty response, or invalid JSON.
    """
    if client is None:
        settings = get_settings()
        client = Groq(api_key=settings.groq_api_key)

    system_prompt = _load_system_prompt(_PLANNER_PROMPT_FILE)
    user_prompt = _build_user_prompt(schema, question)

    if history:
        history_text = "\n".join(f"- Q: {t['question']}\n  A: {t['answer']}" for t in history)
        user_prompt = f"Conversation history:\n{history_text}\n\n{user_prompt}"
    log.info("llm.call.started", fn="generate_code", model=model)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        log.error("llm.call.failed", fn="generate_code", error=str(exc))
        raise LLMError(f"Groq API call failed: {exc}") from exc
    log.info("llm.call.completed", fn="generate_code")
    if not response.choices:
        raise LLMError("LLM returned no choices")

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise LLMError("LLM returned empty content")

    cleaned = _strip_code_fences(content)

    try:
        return PlannerOutput.model_validate_json(cleaned)
    except ValidationError as exc:
        raise LLMError(f"Planner returned invalid structured output: {exc}") from exc


class FixedCode(BaseModel):
    """Structured output from the fixer LLM call."""

    code: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)


def _build_fixer_user_prompt(
    schema: DatasetSchema,
    question: str,
    previous_code: str,
    error_message: str,
    error_traceback: str | None,
    previous_errors: list[str],
) -> str:
    """Compose the fixer user message from failure context."""
    columns_desc = "\n".join(
        f"- {c.name} ({c.dtype}), nulls: {c.null_count}/{schema.row_count}" for c in schema.columns
    )

    parts = [
        f"Question: {question}",
        f"CSV schema:\nfile: {schema.file_name}\nrows: {schema.row_count}\n"
        f"columns:\n{columns_desc}",
        f"Failed code:\n{previous_code}",
        f"Error message:\n{error_message}",
    ]

    if error_traceback:
        parts.append(f"Traceback:\n{error_traceback}")

    if previous_errors:
        earlier = "\n".join(f"- {e}" for e in previous_errors)
        parts.append(f"Earlier errors from previous attempts:\n{earlier}")

    parts.append("Respond with a JSON object.")

    return "\n\n".join(parts)


def fix_code(
    schema: DatasetSchema,
    question: str,
    previous_code: str,
    error_message: str,
    error_traceback: str | None = None,
    previous_errors: list[str] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    client: Groq | None = None,
) -> FixedCode:
    """Ask the LLM to repair code that failed during execution.

    Args:
        schema: The dataset schema.
        question: Original user question.
        previous_code: The code that failed.
        error_message: The exception message.
        error_traceback: The full Python traceback (optional).
        previous_errors: Errors from earlier repair attempts (optional).
        model: Groq model name.
        client: Optional pre-built Groq client (for tests).

    Returns:
        A validated FixedCode object.

    Raises:
        LLMError: On API failure, empty response, or invalid JSON.
    """
    if client is None:
        settings = get_settings()
        client = Groq(api_key=settings.groq_api_key)

    system_prompt = _load_system_prompt(_FIXER_PROMPT_FILE)
    user_prompt = _build_fixer_user_prompt(
        schema=schema,
        question=question,
        previous_code=previous_code,
        error_message=error_message,
        error_traceback=error_traceback,
        previous_errors=previous_errors or [],
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Groq API call failed: {exc}") from exc

    if not response.choices:
        raise LLMError("LLM returned no choices")

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise LLMError("LLM returned empty content")

    cleaned = _strip_code_fences(content)

    try:
        return FixedCode.model_validate_json(cleaned)
    except ValidationError as exc:
        raise LLMError(f"Fixer returned invalid structured output: {exc}") from exc


def _load_system_prompt(name: str = _SYSTEM_PROMPT_FILE) -> str:
    """Load a system prompt file shipped with the package."""
    path = _PROMPTS_DIR / name
    if not path.is_file():
        raise LLMError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _build_user_prompt(schema: DatasetSchema, question: str) -> str:
    """Compose the user message from schema and question."""
    columns_desc = "\n".join(
        f"- {c.name} ({c.dtype}), nulls: {c.null_count}/{schema.row_count}" for c in schema.columns
    )
    samples_desc = "\n".join(str(row) for row in schema.sample_rows)

    return (
        f"CSV schema:\n"
        f"file: {schema.file_name}\n"
        f"rows: {schema.row_count}\n"
        f"columns:\n{columns_desc}\n\n"
        f"sample rows:\n{samples_desc}\n\n"
        f"Question: {question}"
        f"Respond with a JSON object."
    )


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model added them anyway."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def generate_code(
    schema: DatasetSchema,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Groq | None = None,
) -> GeneratedCode:
    """Generate structured pandas code for a question about a CSV schema.

    Args:
        schema: The dataset schema (from ``extract_schema``).
        question: Natural-language question.
        model: Groq model name.
        client: Optional pre-built Groq client (for tests). If None, a new
            client is built using the API key from Settings.

    Returns:
        A validated ``GeneratedCode`` object.

    Raises:
        LLMError: On API failure, empty response, or invalid JSON.
    """
    if client is None:
        settings = get_settings()
        client = Groq(api_key=settings.groq_api_key)

    system_prompt = _load_system_prompt()
    user_prompt = _build_user_prompt(schema, question)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 — provider SDKs raise varied types
        raise LLMError(f"Groq API call failed: {exc}") from exc

    if not response.choices:
        raise LLMError("LLM returned no choices")

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise LLMError("LLM returned empty content")

    cleaned = _strip_code_fences(content)

    try:
        return GeneratedCode.model_validate_json(cleaned)
    except ValidationError as exc:
        raise LLMError(f"LLM returned invalid structured output: {exc}") from exc


class SummaryOutput(BaseModel):
    """Structured output from the summarizer LLM call."""

    answer: str = Field(..., min_length=1)


def _build_summarizer_user_prompt(
    schema: DatasetSchema,
    question: str,
    result_repr: str,
    explanation: str,
) -> str:
    """Compose the summarizer user message."""
    columns_desc = "\n".join(f"- {c.name} ({c.dtype})" for c in schema.columns)

    return (
        f"Question: {question}\n\n"
        f"CSV schema:\nfile: {schema.file_name}\n"
        f"columns:\n{columns_desc}\n\n"
        f"Final result:\n{result_repr}\n\n"
        f"Code explanation:\n{explanation}"
        f"Respond with a JSON object."
    )


def summarize(
    schema: DatasetSchema,
    question: str,
    result_repr: str,
    explanation: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Groq | None = None,
) -> SummaryOutput:
    """Ask the LLM to produce a natural-language answer.

    Args:
        schema: The dataset schema.
        question: Original user question.
        result_repr: String representation of the final result.
        explanation: What the code was doing.
        model: Groq model name.
        client: Optional pre-built Groq client (for tests).

    Returns:
        A validated SummaryOutput.

    Raises:
        LLMError: On API failure, empty response, or invalid JSON.
    """
    if client is None:
        settings = get_settings()
        client = Groq(api_key=settings.groq_api_key)

    system_prompt = _load_system_prompt(_SUMMARIZER_PROMPT_FILE)
    user_prompt = _build_summarizer_user_prompt(
        schema=schema,
        question=question,
        result_repr=result_repr,
        explanation=explanation,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Groq API call failed: {exc}") from exc

    if not response.choices:
        raise LLMError("LLM returned no choices")

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise LLMError("LLM returned empty content")

    cleaned = _strip_code_fences(content)

    try:
        return SummaryOutput.model_validate_json(cleaned)
    except ValidationError as exc:
        raise LLMError(f"Summarizer returned invalid structured output: {exc}") from exc
