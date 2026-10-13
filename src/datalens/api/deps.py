"""FastAPI dependencies (auth, DB, rate limiting).

Auth and rate limiting land in Task 4.5. DB is wired now.
"""

from __future__ import annotations

from datalens.db.session import get_db_session

__all__ = ["get_db_session"]
