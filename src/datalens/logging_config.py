"""Structured logging setup.

In production (DATALENS_LOG_JSON=true), logs are emitted as one JSON
object per line — easy to ship to a log aggregator. In dev, logs are
colorized and human-readable.

Call `configure_logging()` once at process startup (API, CLI).
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(json_output: bool | None = None) -> None:
    """Configure structlog + stdlib logging.

    Args:
        json_output: If None, read from env var DATALENS_LOG_JSON
            (default: False for dev, True in production).
    """
    if json_output is None:
        json_output = os.environ.get("DATALENS_LOG_JSON", "false").lower() == "true"

    # Route stdlib logging (from libraries like httpx, uvicorn) through structlog.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "datalens") -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to a component name."""
    return structlog.get_logger(name)
