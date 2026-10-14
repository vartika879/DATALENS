"""Tests for retry + fallback behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from datalens.llm import (
    LLMError,
    _call_with_fallback,
    _is_retryable_error,
)


def test_is_retryable_error_detects_429() -> None:
    assert _is_retryable_error(Exception("429 rate limit"))
    assert _is_retryable_error(Exception("Too Many Requests"))
    assert _is_retryable_error(Exception("Connection timeout"))
    assert not _is_retryable_error(Exception("Invalid API key"))


def test_primary_succeeds_first_try() -> None:
    calls: list[str] = []

    def primary():
        calls.append("primary")
        return "ok"

    result = _call_with_fallback(primary, None, "test")
    assert result == "ok"
    assert calls == ["primary"]


@patch("datalens.llm.time.sleep")
def test_retries_primary_on_transient(mock_sleep) -> None:
    attempts: list[int] = []

    def primary():
        attempts.append(len(attempts))
        if len(attempts) < 2:
            raise Exception("429 rate limit")
        return "ok"

    result = _call_with_fallback(primary, None, "test")
    assert result == "ok"
    assert len(attempts) == 2
    assert mock_sleep.called


@patch("datalens.llm.time.sleep")
def test_falls_back_when_primary_exhausted(mock_sleep) -> None:
    def primary():
        raise Exception("429 rate limit")

    def fallback():
        return "gemini answer"

    result = _call_with_fallback(primary, fallback, "test")
    assert result == "gemini answer"


@patch("datalens.llm.time.sleep")
def test_both_fail_raises_llm_error(mock_sleep) -> None:
    def primary():
        raise Exception("429 rate limit")

    def fallback():
        raise Exception("503 service unavailable")

    with pytest.raises(LLMError, match="Both Groq and Gemini"):
        _call_with_fallback(primary, fallback, "test")


def test_non_retryable_fails_fast() -> None:
    calls: list[str] = []

    def primary():
        calls.append("primary")
        raise Exception("Invalid API key")

    def fallback():
        calls.append("fallback")
        return "should not reach"

    with pytest.raises(Exception, match="Invalid API key"):
        _call_with_fallback(primary, fallback, "test")
    assert calls == ["primary"]  # no retry, no fallback
