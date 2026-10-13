"""Tests for the Streamlit HTTP client."""

from __future__ import annotations

import httpx
import pytest

from datalens.ui.client import DataLensAPIError, DataLensClient


def _make_client(handler) -> DataLensClient:
    transport = httpx.MockTransport(handler)
    client = DataLensClient(base_url="http://test", api_key="k")
    client._client = httpx.Client(
        base_url="http://test",
        headers={"X-API-Key": "k"},
        transport=transport,
    )
    return client


def test_upload_sends_multipart() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/upload"
        assert request.headers.get("X-API-Key") == "k"
        return httpx.Response(
            200,
            json={
                "dataset_id": "abc",
                "file_name": "t.csv",
                "row_count": 3,
                "column_count": 2,
            },
        )

    client = _make_client(handler)
    resp = client.upload("t.csv", b"a,b\n1,2\n")
    assert resp["dataset_id"] == "abc"


def test_ask_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "run_id": "r1",
                "status": "success",
                "answer": "42",
                "latency_ms": 100.0,
                "retry_count": 0,
            },
        )

    client = _make_client(handler)
    resp = client.ask("s1", "q?")
    assert resp["answer"] == "42"


def test_error_translates_to_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid API key."})

    client = _make_client(handler)
    with pytest.raises(DataLensAPIError) as exc:
        client.health()
    assert exc.value.status_code == 401
    assert "Invalid API key" in exc.value.detail


def test_validation_error_flattens_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"detail": [{"msg": "bad rating"}, {"msg": "missing run_id"}]},
        )

    client = _make_client(handler)
    with pytest.raises(DataLensAPIError) as exc:
        client.feedback("r1", "invalid")
    assert "bad rating" in exc.value.detail
