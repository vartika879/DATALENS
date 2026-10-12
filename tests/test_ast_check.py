"""Tests for the AST-based static analyzer."""

from __future__ import annotations

import pytest

from datalens.ast_check import (
    ASTCheckError,
    assert_code_is_safe,
    check_code,
)

# --- Safe code --------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "result = df['sales'].sum()",
        "result = df.groupby('city')['sales'].mean()",
        "result = len(df)",
        "result = pd.to_datetime(df['date'])",
        "result = np.where(df['sales'] > 100, 'high', 'low')",
        "result = df.head(5).to_dict()",
    ],
)
def test_safe_code_passes(code: str) -> None:
    assert check_code(code) == []
    assert_code_is_safe(code)  # should not raise


# --- Blocked imports --------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = os.getcwd()",
        "import sys\nresult = sys.version",
        "import subprocess\nresult = subprocess.run(['ls'])",
        "from os import getcwd\nresult = getcwd()",
        "from subprocess import run\nresult = 1",
        "import socket\nresult = 1",
    ],
)
def test_blocked_imports(code: str) -> None:
    violations = check_code(code)
    assert any(v.kind == "import" for v in violations)
    with pytest.raises(ASTCheckError, match="import"):
        assert_code_is_safe(code)


# --- Blocked builtins -------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "result = open('/etc/passwd').read()",
        "result = eval('1+1')",
        "result = exec('x=1')",
        "result = compile('1', '<x>', 'eval')",
        "result = __import__('os')",
        "result = getattr(df, 'shape')",
    ],
)
def test_blocked_builtins(code: str) -> None:
    violations = check_code(code)
    assert len(violations) >= 1
    with pytest.raises(ASTCheckError):
        assert_code_is_safe(code)


# --- Blocked attributes -----------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "result = ().__class__.__bases__",
        "result = object.__subclasses__()",
        "result = (1).__class__.__mro__",
        "result = df.__dict__",
        "result = __builtins__",
    ],
)
def test_blocked_attributes(code: str) -> None:
    violations = check_code(code)
    assert len(violations) >= 1
    with pytest.raises(ASTCheckError):
        assert_code_is_safe(code)


# --- Suspicious path strings ------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "path = '/etc/passwd'\nresult = 1",
        "path = 'C:/Users/Admin/secrets.txt'\nresult = 1",
        "path = '.ssh/id_rsa'\nresult = 1",
    ],
)
def test_suspicious_path_strings(code: str) -> None:
    violations = check_code(code)
    assert any(v.kind == "path" for v in violations)


# --- Syntax errors ----------------------------------------------------------


def test_syntax_error_raises() -> None:
    with pytest.raises(SyntaxError):
        check_code("result = df[")


# --- Violation carries line number ------------------------------------------


def test_violation_has_line_number() -> None:
    violations = check_code("x = 1\ny = 2\nimport os\nresult = 1")
    assert len(violations) == 1
    assert violations[0].line == 3
    assert violations[0].kind == "import"
