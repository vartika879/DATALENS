"""Tests for /history endpoint."""

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


def _upload(client: TestClient) -> str:
    resp = client.post(
        "/upload",
        files={"file": ("t.csv", io.BytesIO(b"city,sales\nA,10\nB,20\n"), "text/csv")},
    )
    return resp.json()["dataset_id"]


def _session(client: TestClient, dataset_id: str) -> str:
    resp = client.post("/session", json={"dataset_id": dataset_id})
    return resp.json()["session_id"]


@patch("datalens.pipeline.generate_code")
def test_history_empty_session(mock_gen, client: TestClient) -> None:
    did = _upload(client)
    sid = _session(client, did)
    resp = client.get(f"/history/{sid}")
    assert resp.status_code == 200
    assert resp.json()["runs"] == []


@patch("datalens.pipeline.generate_code")
def test_history_returns_runs(mock_gen, client: TestClient) -> None:
    mock_gen.return_value = GeneratedCode(
        code="result = df['sales'].sum()",
        explanation="sum",
        self_reported_confidence=0.9,
    )
    did = _upload(client)
    sid = _session(client, did)

    client.post("/ask", json={"session_id": sid, "question": "total?"})
    client.post("/ask", json={"session_id": sid, "question": "again?"})

    resp = client.get(f"/history/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == sid
    assert len(body["runs"]) == 2
    assert body["runs"][0]["question"] == "total?"
    assert body["runs"][1]["question"] == "again?"


def test_history_unknown_session(client: TestClient) -> None:
    resp = client.get("/history/nope")
    assert resp.status_code == 404


def test_history_requires_api_key() -> None:
    c = TestClient(create_app())  # no key
    resp = c.get("/history/whatever")
    assert resp.status_code == 401
