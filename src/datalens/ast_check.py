"""Static analysis of LLM-generated code before execution.

This module provides the FIRST layer of defense. It is not a security
boundary — a determined attacker can craft code that passes AST checks
but escapes at runtime. Its job is to catch obvious, cheap-to-detect
patterns (``import os``, ``open(...)``, ``eval(...)``) quickly and
consistently, so that the Docker sandbox only has to handle the rare,
subtle cases.

Layers of defense:
    1. AST checks (this module) — fast, cheap, catches obvious issues.
    2. Docker sandbox — real isolation for anything that passes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# --- Blocklists -------------------------------------------------------------

BLOCKED_IMPORTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pathlib",
        "importlib",
        "ctypes",
        "pickle",
        "marshal",
        "builtins",
        "io",
        "tempfile",
        "glob",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "telnetlib",
        "smtplib",
        "asyncio",
    }
)

BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "memoryview",
    }
)

BLOCKED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__class__",
        "__bases__",
        "__mro__",
        "__subclasses__",
        "__globals__",
        "__code__",
        "__dict__",
        "__builtins__",
        "__loader__",
        "__spec__",
    }
)

BLOCKED_PATH_PATTERNS: tuple[str, ...] = (
    "/etc/",
    "/root/",
    "/home/",
    "/var/",
    "C:\\",
    "C:/",
    "/proc/",
    "/sys/",
    ".ssh",
    ".aws",
    ".env",
    "id_rsa",
    "credentials",
)


# --- Result model -----------------------------------------------------------


@dataclass
class ASTViolation:
    """A single AST-level violation."""

    line: int
    kind: str  # "import" | "builtin" | "attribute" | "path"
    detail: str


class ASTCheckError(Exception):
    """Raised when AST analysis finds a violation."""

    def __init__(self, violations: list[ASTViolation]) -> None:
        self.violations = violations
        message = "Code rejected by static analysis:\n" + "\n".join(
            f"  line {v.line}: [{v.kind}] {v.detail}" for v in violations
        )
        super().__init__(message)


# --- Visitor ----------------------------------------------------------------


class _SecurityVisitor(ast.NodeVisitor):
    """Walks the AST and collects security violations."""

    def __init__(self) -> None:
        self.violations: list[ASTViolation] = []

    # --- Imports ------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in BLOCKED_IMPORTS:
                self.violations.append(
                    ASTViolation(
                        line=node.lineno,
                        kind="import",
                        detail=f"import of '{alias.name}' is blocked",
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in BLOCKED_IMPORTS:
                self.violations.append(
                    ASTViolation(
                        line=node.lineno,
                        kind="import",
                        detail=f"from-import of '{node.module}' is blocked",
                    )
                )
        self.generic_visit(node)

    # --- Calls to blocked builtins -----------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # Direct name: open(...), eval(...)
        if isinstance(func, ast.Name) and func.id in BLOCKED_BUILTINS:
            self.violations.append(
                ASTViolation(
                    line=node.lineno,
                    kind="builtin",
                    detail=f"call to '{func.id}()' is blocked",
                )
            )
        # Attribute access: obj.__class__(), something.eval(...)
        if isinstance(func, ast.Attribute) and func.attr in BLOCKED_ATTRIBUTES:
            self.violations.append(
                ASTViolation(
                    line=node.lineno,
                    kind="attribute",
                    detail=f"call through '{func.attr}' is blocked",
                )
            )
        self.generic_visit(node)

    # --- Attribute access ---------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BLOCKED_ATTRIBUTES:
            self.violations.append(
                ASTViolation(
                    line=node.lineno,
                    kind="attribute",
                    detail=f"attribute access '{node.attr}' is blocked",
                )
            )
        self.generic_visit(node)

    # --- Name access (e.g. bare __builtins__) ------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in {"__builtins__", "__import__"}:
            self.violations.append(
                ASTViolation(
                    line=node.lineno,
                    kind="builtin",
                    detail=f"reference to '{node.id}' is blocked",
                )
            )
        self.generic_visit(node)

    # --- Suspicious string literals ----------------------------------------

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            lowered = node.value.lower()
            for pattern in BLOCKED_PATH_PATTERNS:
                if pattern.lower() in lowered:
                    self.violations.append(
                        ASTViolation(
                            line=node.lineno,
                            kind="path",
                            detail=f"string contains suspicious path '{pattern}'",
                        )
                    )
                    break  # one violation per string is enough
        self.generic_visit(node)


# --- Public API -------------------------------------------------------------


def check_code(code: str) -> list[ASTViolation]:
    """Return a list of AST violations in ``code`` (empty if clean).

    Raises:
        SyntaxError: If the code cannot be parsed.
    """
    tree = ast.parse(code)
    visitor = _SecurityVisitor()
    visitor.visit(tree)
    return visitor.violations


def assert_code_is_safe(code: str) -> None:
    """Raise ASTCheckError if the code contains any violations.

    Raises:
        SyntaxError: If the code cannot be parsed.
        ASTCheckError: If any violations are found.
    """
    violations = check_code(code)
    if violations:
        raise ASTCheckError(violations)
