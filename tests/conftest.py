"""Shared pytest fixtures for the DataLens test suite.

Sets a dummy GROQ_API_KEY so tests don't need a real .env file.
Tests that hit Groq always mock the client — the key is never used.
"""

from __future__ import annotations

import pytest

from datalens.config import get_settings


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a dummy API key and clear the settings cache for each test."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used-mocked-in-tests")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()