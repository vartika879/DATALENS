"""Application configuration loaded from environment variables.

Secrets are loaded from a `.env` file (never hardcoded). The Settings
object is validated by pydantic-settings, so a missing or malformed key
fails loudly at startup rather than at first API call.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Reads from environment variables and `.env`. Unknown variables are
    ignored, so unrelated env vars won't break startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: str = Field(..., min_length=10)
    use_sandbox: bool = False
    datalens_api_key: str = Field(..., min_length=8)
    datalens_rate_limit_per_minute: int = 60
    datalens_api_url: str = "http://localhost:8000"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Cached so that reading `.env` happens only once per process. Tests can
    clear the cache with `get_settings.cache_clear()`.
    """
    return Settings()  # type: ignore[call-arg]
