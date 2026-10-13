"""Tests for /session and /ask endpoints (LLM mocked)."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, select

from datalens.api.app import create_app
from datalens.db.engine import get_engine
from datalens.db.models import Run
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
    return TestClient(create_app())


@pytest.fixture
def uploaded_dataset(client: TestClient) -> str:
    csv = "city,sales\nMumbai,100\nDelhi,200\n"
    resp = client.post(
        "/upload",
        files={"file": ("t.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    return resp.json()["dataset_id"]


def test_create_session(client: TestClient, uploaded_dataset: str) -> None:
    resp = client.post("/session", json={"dataset_id": uploaded_dataset})
    assert resp.status_code == 200
    assert resp.json()["session_id"]


def test_create_session_unknown_dataset(client: TestClient) -> None:
    resp = client.post("/session", json={"dataset_id": "does-not-exist"})
    assert resp.status_code == 404


@patch("datalens.pipeline.generate_code")
def test_ask_success(mock_gen, client: TestClient, uploaded_dataset: str) -> None:
    mock_gen.return_value = GeneratedCode(
        code="result = df['sales'].sum()",
        explanation="sum of sales",
        self_reported_confidence=0.9,
    )
    sid = client.post("/session", json={"dataset_id": uploaded_dataset}).json()["session_id"]
    resp = client.post("/ask", json={"session_id": sid, "question": "total?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["answer"] == "300"
    assert body["code"] == "result = df['sales'].sum()"

    with Session(get_engine()) as s:
        runs = s.exec(select(Run)).all()
        assert len(runs) == 1
        assert runs[0].question == "total?"


def test_ask_unknown_session(client: TestClient) -> None:
    resp = client.post("/ask", json={"session_id": "nope", "question": "?"})
    assert resp.status_code == 404
