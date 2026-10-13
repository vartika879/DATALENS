"""Tests for /feedback endpoint."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from datalens.api.app import create_app
from datalens.db.engine import get_engine
from datalens.llm import GeneratedCode


@pytest.fixture(autouse=True)
def _in_memory_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATALENS_DB_URL", "sqlite:///:memory:")
    get_engine.cache_clear()
    from datalens.db import models  # noqa: F401

    SQLModel.metadata.create_all(get_engine())
    yield
    get_engine.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(
        create_app(),
        headers={"X-API-Key": "test-api-key-12345678"},
    )


def _setup_run(client: TestClient) -> str:
    """Create a dataset, session, and one completed run. Returns run_id."""
    with patch("datalens.pipeline.generate_code") as mock_gen:
        mock_gen.return_value = GeneratedCode(
            code="result = 1",
            explanation="one",
            self_reported_confidence=0.9,
        )
        did = client.post(
            "/upload",
            files={"file": ("t.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
        ).json()["dataset_id"]
        sid = client.post("/session", json={"dataset_id": did}).json()["session_id"]
        return client.post("/ask", json={"session_id": sid, "question": "q"}).json()["run_id"]


def test_feedback_up(client: TestClient) -> None:
    run_id = _setup_run(client)
    resp = client.post(
        "/feedback",
        json={"run_id": run_id, "rating": "up", "comment": "nice"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["rating"] == "up"
    assert body["run_id"] == run_id


def test_feedback_duplicate_returns_409(client: TestClient) -> None:
    run_id = _setup_run(client)
    client.post("/feedback", json={"run_id": run_id, "rating": "up"})
    resp = client.post("/feedback", json={"run_id": run_id, "rating": "down"})
    assert resp.status_code == 409


def test_feedback_invalid_rating(client: TestClient) -> None:
    resp = client.post(
        "/feedback",
        json={"run_id": "any", "rating": "maybe"},
    )
    assert resp.status_code == 422  # pydantic validation


def test_feedback_unknown_run(client: TestClient) -> None:
    resp = client.post("/feedback", json={"run_id": "nope", "rating": "up"})
    assert resp.status_code == 404


def test_feedback_requires_api_key() -> None:
    c = TestClient(create_app())
    resp = c.post("/feedback", json={"run_id": "any", "rating": "up"})
    assert resp.status_code == 401
