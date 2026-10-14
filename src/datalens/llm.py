"""LLM client wrapper for Groq (primary) and Gemini (fallback).

All provider-specific code lives here so the rest of the codebase depends
only on the four public functions: ``generate_code``, ``plan_question``,
``fix_code``, ``summarize``. Each one:

- Loads a system prompt from ``src/datalens/prompts/``.
- Calls Groq with retries and exponential backoff on transient errors.
- Falls back to Gemini (OpenAI-compatible endpoint) if Groq is exhausted.
- Validates the JSON response with Pydantic.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from groq import Groq
from pydantic import BaseModel, Field, ValidationError, model_validator

from datalens.config import get_settings
from datalens.logging_config import get_logger
from datalens.schema import DatasetSchema

log = get_logger("datalens.llm")


# --- Constants --------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-oss-20b"

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_FILE = "generate_code_v1.txt"
_PLANNER_PROMPT_FILE = "planner_v2.txt"
_FIXER_PROMPT_FILE = "fixer_v1.txt"
_SUMMARIZER_PROMPT_FILE = "summarizer_v1.txt"

MAX_LLM_RETRIES = 3
RETRY_BASE_DELAY_S = 2.0
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_JSON_HINT = "Respond with a JSON object."


# --- Errors -----------------------------------------------------------------


class LLMError(Exception):
    """Raised when the LLM call fails or the response cannot be parsed."""


# --- Models -----------------------------------------------------------------


class GeneratedCode(BaseModel):
    """Structured output from the LLM.

    ``self_reported_confidence`` is the model's own estimate and is NOT a
    calibrated probability. Do not use it as ground truth.
    """

    code: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)
    self_reported_confidence: float = Field(..., ge=0.0, le=1.0)


PlannerDecision = Literal["proceed", "clarify", "refuse"]


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


class FixedCode(BaseModel):
    """Structured output from the fixer LLM call."""

    code: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)


class SummaryOutput(BaseModel):
    """Structured output from the summarizer LLM call."""

    answer: str = Field(..., min_length=1)


# --- Retry / fallback -------------------------------------------------------


def _is_retryable_error(exc: Exception) -> bool:
    """Detect transient errors that warrant a retry with backoff."""
    msg = str(exc).lower()
    return any(
        m in msg
        for m in (
            "429",
            "rate limit",
            "too many requests",
            "quota",
            "timeout",
            "connection",
            "502",
            "503",
            "504",
            "temporarily",
        )
    )


def _get_gemini_client() -> Any | None:
    """Return an OpenAI-compatible Gemini client, or None if not configured."""
    settings = get_settings()
    if not settings.gemini_api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return OpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_BASE_URL)


def _call_with_fallback(
    primary_call: Callable[[], Any],
    fallback_call: Callable[[], Any] | None,
    fn_name: str,
) -> Any:
    """Retry primary with exponential backoff; on exhaustion, use fallback.

    Non-retryable errors (auth, bad request) fail fast — no retries, no
    fallback. Only rate limits / transient errors trigger retry or fallback.
    """
    last_exc: Exception | None = None

    for attempt in range(MAX_LLM_RETRIES):
        try:
            return primary_call()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_retryable_error(exc):
                raise
            if attempt < MAX_LLM_RETRIES - 1:
                delay = RETRY_BASE_DELAY_S * (2**attempt)
                log.warning(
                    "llm.retry",
                    fn=fn_name,
                    attempt=attempt + 1,
                    delay_s=delay,
                    error=str(exc)[:120],
                )
                time.sleep(delay)

    if fallback_call is None:
        assert last_exc is not None
        raise last_exc

    log.warning("llm.fallback", fn=fn_name, provider="gemini")
    try:
        return fallback_call()
    except Exception as fallback_exc:  # noqa: BLE001
        log.error(
            "llm.fallback_failed",
            fn=fn_name,
            error=str(fallback_exc)[:120],
        )
        raise LLMError(f"Both Groq and Gemini fallback failed: {fallback_exc}") from fallback_exc


# --- Prompt / response helpers ----------------------------------------------


def _load_system_prompt(name: str = _SYSTEM_PROMPT_FILE) -> str:
    """Load a system prompt file shipped with the package."""
    path = _PROMPTS_DIR / name
    if not path.is_file():
        raise LLMError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


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
        f"Question: {question}\n\n"
        f"{_JSON_HINT}"
    )


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
    parts.append(_JSON_HINT)
    return "\n\n".join(parts)


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
        f"Code explanation:\n{explanation}\n\n"
        f"{_JSON_HINT}"
    )


# --- Internals: build the two call closures ---------------------------------


def _build_call_pair(
    client: Groq,
    model: str,
    messages: list[dict],
) -> tuple[Callable[[], Any], Callable[[], Any] | None]:
    """Return (groq_call, gemini_call_or_none) closures."""

    def _groq_call() -> Any:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

    gemini = _get_gemini_client()
    if gemini is None:
        return _groq_call, None

    gemini_model = get_settings().gemini_model

    def _gemini_call() -> Any:
        return gemini.chat.completions.create(
            model=gemini_model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

    return _groq_call, _gemini_call


def _call_llm(
    client: Groq | None,
    model: str,
    messages: list[dict],
    fn_name: str,
) -> Any:
    """Make the LLM call with retry + fallback. Returns the raw response."""
    if client is None:
        settings = get_settings()
        client = Groq(api_key=settings.groq_api_key)

    primary, fallback = _build_call_pair(client, model, messages)

    log.info("llm.call.started", fn=fn_name, model=model)
    try:
        return _call_with_fallback(primary, fallback, fn_name)
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.error("llm.call.failed", fn=fn_name, error=str(exc)[:200])
        raise LLMError(f"LLM call failed: {exc}") from exc
    finally:
        log.info("llm.call.completed", fn=fn_name)


def _extract_content(response: Any) -> str:
    """Validate and return the first choice's content."""
    if not response.choices:
        raise LLMError("LLM returned no choices")
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise LLMError("LLM returned empty content")
    return content


# --- Public API -------------------------------------------------------------


def generate_code(
    schema: DatasetSchema,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Groq | None = None,
) -> GeneratedCode:
    """Generate structured pandas code for a question about a CSV schema."""
    messages = [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user", "content": _build_user_prompt(schema, question)},
    ]
    response = _call_llm(client, model, messages, "generate_code")
    content = _strip_code_fences(_extract_content(response))
    try:
        return GeneratedCode.model_validate_json(content)
    except ValidationError as exc:
        raise LLMError(f"LLM returned invalid structured output: {exc}") from exc


def plan_question(
    schema: DatasetSchema,
    question: str,
    history: list[dict] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    client: Groq | None = None,
) -> PlannerOutput:
    """Ask the LLM how to handle a question against a schema."""
    user_prompt = _build_user_prompt(schema, question)
    if history:
        history_text = "\n".join(f"- Q: {t['question']}\n  A: {t['answer']}" for t in history)
        user_prompt = f"Conversation history:\n{history_text}\n\n{user_prompt}"

    messages = [
        {"role": "system", "content": _load_system_prompt(_PLANNER_PROMPT_FILE)},
        {"role": "user", "content": user_prompt},
    ]
    response = _call_llm(client, model, messages, "plan_question")
    content = _strip_code_fences(_extract_content(response))
    try:
        return PlannerOutput.model_validate_json(content)
    except ValidationError as exc:
        raise LLMError(f"Planner returned invalid structured output: {exc}") from exc


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
    """Ask the LLM to repair code that failed during execution."""
    messages = [
        {"role": "system", "content": _load_system_prompt(_FIXER_PROMPT_FILE)},
        {
            "role": "user",
            "content": _build_fixer_user_prompt(
                schema=schema,
                question=question,
                previous_code=previous_code,
                error_message=error_message,
                error_traceback=error_traceback,
                previous_errors=previous_errors or [],
            ),
        },
    ]
    response = _call_llm(client, model, messages, "fix_code")
    content = _strip_code_fences(_extract_content(response))
    try:
        return FixedCode.model_validate_json(content)
    except ValidationError as exc:
        raise LLMError(f"Fixer returned invalid structured output: {exc}") from exc


def summarize(
    schema: DatasetSchema,
    question: str,
    result_repr: str,
    explanation: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Groq | None = None,
) -> SummaryOutput:
    """Ask the LLM to produce a natural-language answer."""
    messages = [
        {"role": "system", "content": _load_system_prompt(_SUMMARIZER_PROMPT_FILE)},
        {
            "role": "user",
            "content": _build_summarizer_user_prompt(
                schema=schema,
                question=question,
                result_repr=result_repr,
                explanation=explanation,
            ),
        },
    ]
    response = _call_llm(client, model, messages, "summarize")
    content = _strip_code_fences(_extract_content(response))
    try:
        return SummaryOutput.model_validate_json(content)
    except ValidationError as exc:
        raise LLMError(f"Summarizer returned invalid structured output: {exc}") from exc
