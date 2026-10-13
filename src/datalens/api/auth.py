"""API key authentication for the DataLens API.

Simple header-based auth: clients send ``X-API-Key: <key>``. The key is
loaded from settings (which reads ``.env``). Constant-time comparison
prevents timing attacks.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from datalens.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    """FastAPI dependency that validates the X-API-Key header.

    Returns the validated key on success, raises 401 otherwise.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    settings = get_settings()
    if not hmac.compare_digest(x_api_key, settings.datalens_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key
