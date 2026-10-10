"""Tests for the Groq LLM wrapper.

Real API calls are never made here — the Groq client is mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datalens.llm import (
    SYSTEM_PROMPT,
    LLMError,
    _build_user_prompt,
    _strip_code_fences,
    generate_code,
)
from datalens.schema import ColumnInfo, DatasetSchema


@pytest.fixture
def sample_schema() -> DatasetSchema:
    return DatasetSchema(
        file_name="sample.csv",
        row_count=5,
        column_count=2,
        columns=[
            ColumnInfo(name="city", dtype="object", null_count=0, null_percentage=0.0),
            ColumnInfo(name="sales", dtype="int64", null_count=1, null_percentage=20.0),
        ],
        sample_rows=[{"city": "Mumbai", "sales": 100}],
    )


def _fake_response(content: str) -> SimpleNamespace:
    """Mimic the shape of a Groq chat completion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


# --- _build_user_prompt -----------------------------------------------------


def test_build_user_prompt_includes_schema_and_question(
    sample_schema: DatasetSchema,
) -> None:
    prompt = _build_user_prompt(sample_schema, "total sales per city?")
    assert "city (object)" in prompt
    assert "sales (int64)" in prompt
    assert "total sales per city?" in prompt
    assert "sample.csv" in prompt


# --- _strip_code_fences -----------------------------------------------------


def test_strip_code_fences_removes_python_fence() -> None:
    raw = "```python\ndf.groupby('city').sum()\n```"
    assert _strip_code_fences(raw) == "df.groupby('city').sum()"


def test_strip_code_fences_leaves_plain_code_alone() -> None:
    raw = "df.groupby('city').sum()"
    assert _strip_code_fences(raw) == raw


# --- generate_code (mocked client) ------------------------------------------


def test_generate_code_returns_model_content(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response("result = df['sales'].sum()")

    code = generate_code(sample_schema, "total sales?", client=client)

    assert code == "result = df['sales'].sum()"
    client.chat.completions.create.assert_called_once()
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["messages"][0]["content"] == SYSTEM_PROMPT


def test_generate_code_strips_fences(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(
        "```python\nresult = df.shape[0]\n```"
    )

    code = generate_code(sample_schema, "how many rows?", client=client)
    assert code == "result = df.shape[0]"


def test_generate_code_raises_on_empty_content(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response("")

    with pytest.raises(LLMError, match="empty content"):
        generate_code(sample_schema, "anything", client=client)


def test_generate_code_raises_on_no_choices(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])

    with pytest.raises(LLMError, match="no choices"):
        generate_code(sample_schema, "anything", client=client)


def test_generate_code_wraps_api_errors(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network boom")

    with pytest.raises(LLMError, match="Groq API call failed"):
        generate_code(sample_schema, "anything", client=client)
