"""FastAPI application factory.

The API is intentionally separate from the Streamlit UI so that:
- the UI can be swapped (or removed) without touching business logic
- the API can be consumed by other clients
- the agent runs as a service, not embedded in a UI framework
"""

from __future__ import annotations

from fastapi import FastAPI

from datalens import __version__
from datalens.api.routes import health


def create_app() -> FastAPI:
    """Build and return a FastAPI application instance."""
    app = FastAPI(
        title="DataLens API",
        version=__version__,
        description="Agentic natural-language data analysis over CSV files.",
    )

    app.include_router(health.router)

    return app
