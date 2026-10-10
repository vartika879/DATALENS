"""Smoke test: verify the package is importable and versioned."""

import datalens


def test_package_version_is_defined() -> None:
    assert datalens.__version__ == "0.0.1"
