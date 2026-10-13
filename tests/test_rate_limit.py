"""Tests for the in-memory rate limiter."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from datalens.api.rate_limit import RateLimiter


def test_under_limit_allows() -> None:
    limiter = RateLimiter(max_per_minute=3)
    for _ in range(3):
        limiter.check("k")  # should not raise


def test_over_limit_raises_429() -> None:
    limiter = RateLimiter(max_per_minute=2)
    limiter.check("k")
    limiter.check("k")
    with pytest.raises(HTTPException) as exc:
        limiter.check("k")
    assert exc.value.status_code == 429


def test_separate_keys_independent() -> None:
    limiter = RateLimiter(max_per_minute=1)
    limiter.check("k1")
    limiter.check("k2")  # different key — allowed
    with pytest.raises(HTTPException):
        limiter.check("k1")


def test_reset_clears_buckets() -> None:
    limiter = RateLimiter(max_per_minute=1)
    limiter.check("k")
    limiter.reset()
    limiter.check("k")  # allowed again
