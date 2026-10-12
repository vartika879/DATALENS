"""Shared pytest fixtures for the DataLens test suite.

Unit tests mock the Groq client, so any non-empty API key works. We set a
dummy key for those tests.

Slow tests (marked ``@pytest.mark.slow``) call the real Groq API and need
the actual key from `.env`. We do NOT override the key for those, so that
pydantic-settings loads it from the file.
"""

from __future__ import annotations

import pytest

from datalens.config import get_settings


@pytest.fixture(autouse=True)
def _test_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a dummy API key for unit tests; skip slow tests.

    Slow tests need the real key from `.env`. Overriding it with a dummy
    value causes 401 errors from the Groq API.
    """
    if request.node.get_closest_marker("slow") is None:
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used-mocked-in-tests")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
