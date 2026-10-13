"""Tests for /upload endpoint."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, select

from datalens.api.app import create_app
from datalens.db.engine import get_engine
from datalens.db.models import Dataset


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


def _csv_file(content: str, name: str = "test.csv") -> dict:
    return {"file": (name, io.BytesIO(content.encode("utf-8")), "text/csv")}


def test_upload_valid_csv(client: TestClient) -> None:
    csv = "city,sales\nMumbai,100\nDelhi,200\n"
    response = client.post("/upload", files=_csv_file(csv))
    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 2
    assert body["column_count"] == 2
    assert body["dataset_id"]

    with Session(get_engine()) as s:
        rows = s.exec(select(Dataset)).all()
        assert len(rows) == 1
        assert rows[0].id == body["dataset_id"]


def test_upload_rejects_non_csv(client: TestClient) -> None:
    response = client.post(
        "/upload",
        files={"file": ("data.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/upload",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert response.status_code == 400
