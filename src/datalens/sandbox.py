"""Host-side wrapper for the Docker sandbox.

Runs generated pandas code inside an isolated container with:
- no network
- memory/CPU limits
- read-only filesystem (with a small tmpfs at /tmp)
- non-root user
- timeout

SECURITY WARNING
----------------
The Docker sandbox provides a real isolation boundary — unlike the direct
``exec`` fallback in ``executor.py``. However, running Docker-in-Docker or
mounting the host Docker socket has its own risks that are documented in
the threat model (Phase 3, Task 5). Do not treat this as absolute security.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SANDBOX_IMAGE = "datalens-sandbox:latest"
DEFAULT_TIMEOUT_SECONDS = 15
MEMORY_LIMIT = "512m"
CPU_LIMIT = "1"


class SandboxError(Exception):
    """Raised when the sandbox cannot be invoked at all."""


@dataclass
class SandboxResult:
    """Outcome of a sandboxed execution."""

    status: str  # "success" | "error" | "timeout"
    result_type: str | None = None
    result_data: Any = None
    result_repr: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    duration_ms: float = 0.0


def sandbox_image_exists(image: str = SANDBOX_IMAGE) -> bool:
    """Return True if the sandbox image is present locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_in_sandbox(
    code: str,
    csv_path: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    image: str = SANDBOX_IMAGE,
) -> SandboxResult:
    """Execute pandas code inside the Docker sandbox.

    Args:
        code: Python source that assigns the answer to a variable ``result``.
        csv_path: Host path to the CSV (mounted read-only at /data/input.csv).
        timeout: Maximum wall-clock seconds before the container is killed.
        image: Sandbox image name.

    Returns:
        SandboxResult — success with serialized data, or error/timeout.

    Raises:
        SandboxError: If the docker command itself cannot run.
    """
    resolved_csv = Path(csv_path).resolve()
    if not resolved_csv.is_file():
        raise SandboxError(f"CSV not found: {resolved_csv}")

    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--memory",
        MEMORY_LIMIT,
        "--cpus",
        CPU_LIMIT,
        "--read-only",
        "--tmpfs",
        "/tmp:size=64m",
        "--user",
        "sandbox",
        "--workdir",
        "/sandbox",
        "-v",
        f"{resolved_csv}:/data/input.csv:ro",
        image,
    ]

    payload = json.dumps({"code": code})
    started = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            status="timeout",
            error_type="TimeoutError",
            error_message=f"Sandbox exceeded {timeout}s timeout",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
    except FileNotFoundError as exc:
        raise SandboxError("docker CLI not found on PATH") from exc

    duration_ms = (time.perf_counter() - started) * 1000.0

    if result.returncode != 0:
        return SandboxResult(
            status="error",
            error_type="ContainerError",
            error_message=(result.stderr or "container exited with no stderr").strip(),
            duration_ms=duration_ms,
        )

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return SandboxResult(
            status="error",
            error_type="ProtocolError",
            error_message="Sandbox produced non-JSON output",
            error_traceback=result.stdout[:1000],
            duration_ms=duration_ms,
        )

    return SandboxResult(
        status=parsed.get("status", "error"),
        result_type=parsed.get("result_type"),
        result_data=parsed.get("result_data"),
        result_repr=parsed.get("result_repr"),
        error_type=parsed.get("error_type"),
        error_message=parsed.get("error_message"),
        error_traceback=parsed.get("error_traceback"),
        duration_ms=parsed.get("duration_ms", duration_ms),
    )
