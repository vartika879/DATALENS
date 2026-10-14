"""FastAPI application factory.

The API is intentionally separate from the Streamlit UI so that:
- the UI can be swapped (or removed) without touching business logic
- the API can be consumed by other clients
- the agent runs as a service, not embedded in a UI framework
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from datalens import __version__
from datalens.api.routes import ask, feedback, health, history, session, upload
from datalens.db.engine import create_db_and_tables
from datalens.logging_config import configure_logging


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ensure DB tables exist on startup."""
    configure_logging()
    create_db_and_tables()
    yield


def create_app() -> FastAPI:
    """Build and return a FastAPI application instance."""
    app = FastAPI(
        title="DataLens API",
        version=__version__,
        description="Agentic natural-language data analysis over CSV files.",
        lifespan=_lifespan,
    )

    app.include_router(health.router)
    app.include_router(upload.router)
    app.include_router(session.router)
    app.include_router(ask.router)
    app.include_router(history.router)
    app.include_router(feedback.router)
    return app
