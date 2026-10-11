"""Tests for the Groq LLM wrapper (structured output).

Real API calls are never made — the Groq client is mocked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datalens.llm import (
    GeneratedCode,
    LLMError,
    PlannerOutput,
    _build_user_prompt,
    _load_system_prompt,
    _strip_code_fences,
    generate_code,
    plan_question,
)
from datalens.schema import ColumnInfo, DatasetSchema

# --- Fixtures ---------------------------------------------------------------


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


def _valid_payload() -> str:
    return json.dumps(
        {
            "code": "result = df['sales'].mean()",
            "explanation": "Mean of sales column",
            "self_reported_confidence": 0.9,
        }
    )


def _fake_response(content: str) -> SimpleNamespace:
    """Mimic the shape of a Groq chat completion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


# --- _load_system_prompt ----------------------------------------------------


def test_load_system_prompt_returns_non_empty_text() -> None:
    text = _load_system_prompt()
    assert "pandas" in text.lower()
    assert "json" in text.lower()
    assert len(text) > 100


def test_load_system_prompt_raises_on_missing_file() -> None:
    with pytest.raises(LLMError, match="Prompt file not found"):
        _load_system_prompt("does_not_exist.txt")


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


def test_strip_code_fences_removes_json_fence() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert _strip_code_fences(raw) == '{"a": 1}'


def test_strip_code_fences_leaves_plain_json_alone() -> None:
    raw = '{"a": 1}'
    assert _strip_code_fences(raw) == raw


# --- generate_code (mocked client) ------------------------------------------


def test_generate_code_returns_generated_code_model(
    sample_schema: DatasetSchema,
) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(_valid_payload())

    result = generate_code(sample_schema, "mean sales?", client=client)

    assert isinstance(result, GeneratedCode)
    assert result.code == "result = df['sales'].mean()"
    assert result.explanation == "Mean of sales column"
    assert result.self_reported_confidence == pytest.approx(0.9)


def test_generate_code_requests_json_response_format(
    sample_schema: DatasetSchema,
) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(_valid_payload())

    generate_code(sample_schema, "any", client=client)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0.0
    assert kwargs["model"] == "openai/gpt-oss-20b"


def test_generate_code_strips_fences_before_parsing(
    sample_schema: DatasetSchema,
) -> None:
    client = MagicMock()
    fenced = f"```json\n{_valid_payload()}\n```"
    client.chat.completions.create.return_value = _fake_response(fenced)

    result = generate_code(sample_schema, "any", client=client)
    assert result.code == "result = df['sales'].mean()"


def test_generate_code_raises_on_invalid_json(
    sample_schema: DatasetSchema,
) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response("not json at all")

    with pytest.raises(LLMError, match="invalid structured output"):
        generate_code(sample_schema, "any", client=client)


def test_generate_code_raises_on_missing_field(
    sample_schema: DatasetSchema,
) -> None:
    client = MagicMock()
    partial = json.dumps({"code": "result = 1"})
    client.chat.completions.create.return_value = _fake_response(partial)

    with pytest.raises(LLMError, match="invalid structured output"):
        generate_code(sample_schema, "any", client=client)


def test_generate_code_raises_on_confidence_out_of_range(
    sample_schema: DatasetSchema,
) -> None:
    client = MagicMock()
    bad = json.dumps({"code": "x", "explanation": "y", "self_reported_confidence": 1.5})
    client.chat.completions.create.return_value = _fake_response(bad)

    with pytest.raises(LLMError, match="invalid structured output"):
        generate_code(sample_schema, "any", client=client)


def test_generate_code_raises_on_empty_content(
    sample_schema: DatasetSchema,
) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response("")

    with pytest.raises(LLMError, match="empty content"):
        generate_code(sample_schema, "any", client=client)


def test_generate_code_raises_on_no_choices(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])

    with pytest.raises(LLMError, match="no choices"):
        generate_code(sample_schema, "any", client=client)


def test_generate_code_wraps_api_errors(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network boom")

    with pytest.raises(LLMError, match="Groq API call failed"):
        generate_code(sample_schema, "any", client=client)


# --- plan_question (mocked client) ------------------------------------------


def _planner_payload(**overrides) -> str:
    base = {
        "decision": "proceed",
        "plan": "sum the sales column",
        "clarifying_question": "",
        "reasoning": "sales is present",
    }
    base.update(overrides)
    return json.dumps(base)


def test_plan_question_returns_planner_output(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(_planner_payload())

    result = plan_question(sample_schema, "total sales?", client=client)

    assert isinstance(result, PlannerOutput)
    assert result.decision == "proceed"
    assert result.plan == "sum the sales column"


def test_plan_question_clarify_requires_question(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    bad = _planner_payload(decision="clarify", plan="", clarifying_question="")
    client.chat.completions.create.return_value = _fake_response(bad)

    with pytest.raises(LLMError, match="Planner returned invalid"):
        plan_question(sample_schema, "best?", client=client)


def test_plan_question_proceed_requires_plan(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    bad = _planner_payload(decision="proceed", plan="")
    client.chat.completions.create.return_value = _fake_response(bad)

    with pytest.raises(LLMError, match="Planner returned invalid"):
        plan_question(sample_schema, "any", client=client)


def test_plan_question_refuse_ok_without_plan(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    ok = _planner_payload(
        decision="refuse", plan="", clarifying_question="", reasoning="no such column"
    )
    client.chat.completions.create.return_value = _fake_response(ok)

    result = plan_question(sample_schema, "profit?", client=client)
    assert result.decision == "refuse"


def test_plan_question_uses_json_mode(sample_schema: DatasetSchema) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(_planner_payload())
    plan_question(sample_schema, "any", client=client)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
