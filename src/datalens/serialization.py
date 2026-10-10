"""Helpers to convert pandas/numpy values to JSON-safe Python natives.

LLM payloads and API responses must be JSON-serializable. numpy ints,
pandas Timestamps, and NaN all need explicit handling — this module is
the single source of truth for that conversion.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def sanitize_for_json(value: Any) -> Any:
    """Convert a pandas/numpy scalar to a JSON-safe Python native.

    Returns None for NaN. numpy scalars become their Python equivalents.
    Unknown objects fall back to ``str(value)``.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    # pandas Timestamp, Timedelta, Decimal, arbitrary objects.
    return str(value)
