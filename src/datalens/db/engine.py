"""Database engine and schema initialization.

The default engine points at ``data/datalens.db``. Tests can override by
calling ``get_engine.cache_clear()`` and setting ``DATALENS_DB_URL``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, create_engine

DEFAULT_DB_PATH = Path("data") / "datalens.db"


def _default_url() -> str:
    """Return the SQLite URL for the current environment."""
    override = os.environ.get("DATALENS_DB_URL")
    if override:
        return override
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the singleton SQLAlchemy engine."""
    url = _default_url()
    # check_same_thread=False is required for SQLite + FastAPI threadpool.
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, echo=False, connect_args=connect_args)


def create_db_and_tables() -> None:
    """Create all tables if they do not exist. Idempotent."""
    # Import models so SQLModel sees the metadata before create_all runs.
    from datalens.db import models  # noqa: F401

    SQLModel.metadata.create_all(get_engine())
