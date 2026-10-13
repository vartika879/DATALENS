"""HTTP client for the DataLens API.

Wraps httpx.Client with typed methods for each endpoint. Errors are
translated into DataLensAPIError with the server's detail message so the
UI can display something useful.
"""

from __future__ import annotations

from typing import Any

import httpx


class DataLensAPIError(Exception):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class DataLensClient:
    """Thin wrapper around the DataLens HTTP API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key},
            timeout=timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DataLensClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- Internal -----------------------------------------------------------

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            body = response.json()
            detail = body.get("detail", response.text)
        except Exception:
            detail = response.text
        if isinstance(detail, list):
            # FastAPI validation errors come back as a list of dicts.
            detail = "; ".join(str(item) for item in detail)
        raise DataLensAPIError(response.status_code, str(detail))

    # --- Endpoints ----------------------------------------------------------

    def health(self) -> dict[str, Any]:
        r = self._client.get("/health")
        self._raise_for_status(r)
        return r.json()

    def upload(self, filename: str, content: bytes) -> dict[str, Any]:
        files = {"file": (filename, content, "text/csv")}
        r = self._client.post("/upload", files=files)
        self._raise_for_status(r)
        return r.json()

    def create_session(self, dataset_id: str) -> dict[str, Any]:
        r = self._client.post("/session", json={"dataset_id": dataset_id})
        self._raise_for_status(r)
        return r.json()

    def ask(self, session_id: str, question: str) -> dict[str, Any]:
        r = self._client.post(
            "/ask",
            json={"session_id": session_id, "question": question},
        )
        self._raise_for_status(r)
        return r.json()

    def history(self, session_id: str) -> dict[str, Any]:
        r = self._client.get(f"/history/{session_id}")
        self._raise_for_status(r)
        return r.json()

    def feedback(self, run_id: str, rating: str, comment: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"run_id": run_id, "rating": rating}
        if comment:
            payload["comment"] = comment
        r = self._client.post("/feedback", json=payload)
        self._raise_for_status(r)
        return r.json()
