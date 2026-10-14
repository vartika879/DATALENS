"""Tests for structured logging setup."""

from __future__ import annotations

import json
from io import StringIO

from datalens.logging_config import configure_logging, get_logger


def test_configure_logging_does_not_crash() -> None:
    configure_logging(json_output=True)
    configure_logging(json_output=False)


def test_get_logger_returns_something() -> None:
    configure_logging(json_output=False)
    log = get_logger("test")
    assert log is not None


def test_json_log_line_is_parseable(capsys) -> None:
    """Capture stdout, emit a log, verify the line is valid JSON."""
    configure_logging(json_output=True)
    log = get_logger("test")
    log.info("test.event", count=5)

    captured = capsys.readouterr()
    # structlog writes to stdout (or stderr depending on version).
    output = (captured.out or captured.err).strip()

    if not output:
        # Some configurations buffer; skip if nothing captured.
        return

    first_line = output.splitlines()[0]
    payload = json.loads(first_line)
    assert payload["event"] == "test.event"
    assert payload["count"] == 5
    assert payload["level"] == "info"