"""Groq LLM client wrapper.

Produces structured ``GeneratedCode`` objects. All Groq-specific code lives
here so the rest of the codebase depends only on ``generate_code``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from groq import Groq
from pydantic import BaseModel, Field, ValidationError, model_validator

from datalens.config import get_settings
from datalens.schema import DatasetSchema

DEFAULT_MODEL = "openai/gpt-oss-20b"

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_FILE = "generate_code_v1.txt"


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

_PLANNER_PROMPT_FILE = "planner_v1.txt"


class PlannerOutput(BaseModel):
    """Structured output from the planner LLM call."""

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
        return PlannerOutput.model_validate_json(cleaned)
    except ValidationError as exc:
        raise LLMError(f"Planner returned invalid structured output: {exc}") from exc


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
