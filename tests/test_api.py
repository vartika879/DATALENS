"""Tests for the FastAPI app skeleton."""

from __future__ import annotations

from fastapi.testclient import TestClient

from datalens import __version__
from datalens.api.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_health_endpoint_returns_ok() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


def test_openapi_schema_is_available() -> None:
    response = _client().get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "DataLens API"


def test_docs_page_loads() -> None:
    # FastAPI serves Swagger UI by default at /docs
    response = _client().get("/docs")
    assert response.status_code == 200


def test_unknown_route_returns_404() -> None:
    response = _client().get("/does-not-exist")
    assert response.status_code == 404


def test_create_app_returns_independent_instances() -> None:
    a = create_app()
    b = create_app()
    assert a is not b
