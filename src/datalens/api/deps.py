"""FastAPI dependencies (auth, DB, rate limiting)."""

from __future__ import annotations

from datalens.api.auth import require_api_key
from datalens.api.rate_limit import get_rate_limiter
from datalens.db.session import get_db_session

__all__ = ["get_db_session", "get_rate_limiter", "require_api_key"]
