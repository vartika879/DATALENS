"""Tests for API key authentication."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from datalens.api.app import create_app
from datalens.db.engine import get_engine


@pytest.fixture(autouse=True)
def _in_memory_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATALENS_DB_URL", "sqlite:///:memory:")
    get_engine.cache_clear()
    from datalens.db import models  # noqa: F401

    SQLModel.metadata.create_all(get_engine())
    yield
    get_engine.cache_clear()


def _client(api_key: str | None = None) -> TestClient:
    headers = {"X-API-Key": api_key} if api_key else {}
    return TestClient(create_app(), headers=headers)


def _csv() -> dict:
    return {"file": ("t.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")}


def test_upload_without_key_returns_401() -> None:
    resp = _client().post("/upload", files=_csv())
    assert resp.status_code == 401


def test_upload_with_wrong_key_returns_401() -> None:
    resp = _client("wrong-key").post("/upload", files=_csv())
    assert resp.status_code == 401


def test_upload_with_valid_key_returns_200() -> None:
    resp = _client("test-api-key-12345678").post("/upload", files=_csv())
    assert resp.status_code == 200


def test_health_is_public() -> None:
    resp = _client().get("/health")
    assert resp.status_code == 200
