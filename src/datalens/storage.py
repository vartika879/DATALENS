"""Filesystem helpers for uploaded CSVs.

Files are stored under ``data/uploads/`` with a UUID prefix to avoid
collisions. The original filename is preserved (sanitized) for display.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

UPLOAD_DIR = Path("data") / "uploads"


def save_uploaded_csv(original_name: str, content: bytes) -> Path:
    """Save uploaded bytes to a unique path and return it.

    The original filename is sanitized to prevent path traversal. Only the
    basename is kept; directory components are stripped.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Strip any directory components from a hostile filename.
    safe_name = Path(original_name).name or "upload.csv"
    stem = Path(safe_name).stem or "upload"
    suffix = Path(safe_name).suffix or ".csv"

    unique = f"{uuid4().hex}_{stem}{suffix}"
    path = UPLOAD_DIR / unique
    path.write_bytes(content)
    return path
