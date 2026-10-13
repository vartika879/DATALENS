"""Simple in-memory rate limiter keyed by API key.

This is a token-bucket approximation using a sliding window per minute.
It is intentionally simple; production deployments behind multiple
workers should use Redis-backed limiting instead.

Trade-offs:
- In-process only. Multiple workers → separate limits per worker.
- Resets on restart.
- Memory grows with number of keys (fine for small deployments).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, status


class RateLimiter:
    """Sliding-window rate limiter keyed by an arbitrary string (e.g. API key)."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._window = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        """Record a hit for ``key``. Raise 429 if over the limit."""
        now = time.monotonic()
        bucket = self._hits[key]

        # Drop hits older than the window.
        cutoff = now - self._window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= self._max:
            retry_after = int(self._window - (now - bucket[0])) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({self._max}/min).",
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)

    def reset(self) -> None:
        """Clear all buckets. Intended for tests."""
        self._hits.clear()


# Module-level singleton. Configured in app startup.
_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Return the process-wide rate limiter (lazy init)."""
    global _limiter
    if _limiter is None:
        from datalens.config import get_settings

        _limiter = RateLimiter(get_settings().datalens_rate_limit_per_minute)
    return _limiter


def reset_rate_limiter() -> None:
    """Reset the singleton. Intended for tests."""
    global _limiter
    _limiter = None
