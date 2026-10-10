"""Groq LLM client wrapper.

Keeps all Groq-specific code in one place so the rest of the codebase
depends only on `generate_code(schema, question) -> str`. Swapping providers
later means changing only this file.
"""

from __future__ import annotations

from groq import Groq

from datalens.config import get_settings
from datalens.schema import DatasetSchema

DEFAULT_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a pandas code generator.
Given a CSV schema and a user question, respond ONLY with valid Python
pandas code that answers the question.

Rules:
- The DataFrame is already loaded in a variable named `df`.
- The last line must assign the final answer to a variable named `result`.
- Do NOT include any explanation, markdown fences, or imports.
- Only use pandas and numpy. Do not use os, sys, subprocess, or file I/O.
"""


class LLMError(Exception):
    """Raised when the LLM call fails or returns an unusable response."""


def _build_user_prompt(schema: DatasetSchema, question: str) -> str:
    """Compose the user message from schema and question.

    Kept as a separate function so it can be unit-tested without an API call.
    """
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
    """Remove ```python ... ``` fences if the model added them anyway."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # drop first line (```python or ```) and last line (```)
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
) -> str:
    """Generate pandas code for a question about a CSV schema.

    Args:
        schema: The dataset schema (from `extract_schema`).
        question: Natural-language question.
        model: Groq model name.
        client: Optional pre-built Groq client (for tests). If None, a new
            client is built using the API key from Settings.

    Returns:
        The generated pandas code as a plain string.

    Raises:
        LLMError: On API failure or empty/malformed response.
    """
    if client is None:
        settings = get_settings()
        client = Groq(api_key=settings.groq_api_key)

    user_prompt = _build_user_prompt(schema, question)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — provider SDKs raise varied types
        raise LLMError(f"Groq API call failed: {exc}") from exc

    if not response.choices:
        raise LLMError("LLM returned no choices")

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise LLMError("LLM returned empty content")

    return _strip_code_fences(content)
