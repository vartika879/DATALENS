"""Health check endpoint.

Used by:
- deployment platforms (Render, Fly.io) to verify the service is up
- Docker healthchecks
- the CI pipeline as a smoke test
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from datalens import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service liveness and version."""
    return HealthResponse(status="ok", version=__version__)
