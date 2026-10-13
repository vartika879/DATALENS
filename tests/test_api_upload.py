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
    return TestClient(
        create_app(),
        headers={"X-API-Key": "test-api-key-12345678"},
    )


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


def test_upload_rejects_binary_content(client: TestClient) -> None:
    """A .csv with binary bytes should be rejected."""
    binary = b"\x00\x01\x02\x03\xff\xfe\xfd\xfc" * 100
    resp = client.post(
        "/upload",
        files={"file": ("fake.csv", io.BytesIO(binary), "text/csv")},
    )
    assert resp.status_code == 400
    assert "binary" in resp.json()["detail"].lower()


def test_upload_rejects_duplicate_columns(client: TestClient) -> None:
    csv = "a,a,b\n1,2,3\n"
    resp = client.post("/upload", files=_csv_file(csv, name="dup.csv"))
    assert resp.status_code == 400
    assert "duplicate" in resp.json()["detail"].lower()


def test_upload_handles_bom(client: TestClient) -> None:
    """Upload should succeed even with a UTF-8 BOM."""
    csv = "\ufeffname,age\nAlice,30\n"
    resp = client.post("/upload", files=_csv_file(csv, name="bom.csv"))
    assert resp.status_code == 200
    assert resp.json()["column_count"] == 2
