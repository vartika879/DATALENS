# DataLens — Threat Model

**Version:** 1.0
**Phase:** 3 (Secure Execution)
**Date:** 2026-10-12

## Scope

This document covers DataLens's handling of untrusted input: user-uploaded
CSV files and LLM-generated Python code that runs against them.

Out of scope:
- Network-level attacks on the host or Groq API
- Attacks requiring shell access to the deployment host
- The LLM provider's own security posture (Groq)

## Assets

| Asset | Why it matters |
|---|---|
| **User data (uploaded CSVs)** | Contains business/personal information; must not be exfiltrated. |
| **API credentials (GROQ_API_KEY)** | Leaked key → cost abuse, account suspension. |
| **Host filesystem** | `/etc/passwd`, SSH keys, `.env` — must remain unreadable to generated code. |
| **Host process** | The FastAPI/Streamlit process must not be compromised by executed code. |
| **Evaluation data** | Golden dataset answers must remain correct; corrupted results mislead. |

## Threat actors

1. **Curious user** — uploads a benign CSV but asks questions that accidentally trigger risky code paths.
2. **Malicious user** — uploads a CSV with injection payloads, or crafts a question to elicit hostile code from the LLM.
3. **Compromised LLM** — the LLM produces intentionally harmful code (via prompt injection, model poisoning, or provider compromise).

## Threats and mitigations

| # | Threat | Layer | Mitigation | Residual risk |
|---|---|---|---|---|
| T1 | LLM generates `import os; os.system(...)` | AST check | `ast_check.assert_code_is_safe` blocks `os`, `sys`, `subprocess` imports | Bypass via `__import__` — caught by AST too. Encoded/obfuscated — caught by sandbox. |
| T2 | LLM generates `open("/etc/passwd")` | AST + sandbox | AST blocks `open`; sandbox filesystem is read-only except `/tmp` | None known. |
| T3 | LLM-generated `eval()`/`exec()` calls | AST + sandbox | AST blocks `eval`/`exec`; sandbox restricts builtins further | None known. |
| T4 | LLM walks `obj.__class__.__mro__` to escape restricted builtins | AST | AST blocks dunder attribute access (`__class__`, `__mro__`, etc.) | AST is bypassable via `getattr`; sandbox is the real boundary. |
| T5 | LLM sends user data to an external server | Sandbox | `--network none` — no outbound network from sandbox | None known. |
| T6 | Infinite loop / fork bomb | Sandbox | `subprocess.run(timeout=15)` + `--cpus 1` + `--memory 512m` | A CPU-bound loop under 15s can still run; acceptable. |
| T7 | Memory exhaustion via huge DataFrame | Sandbox | `--memory 512m` — OOM kills the container | CSV itself may fill memory before exec; file size limit (20 MB) caps this. |
| T8 | Prompt injection via CSV cell/column/sample row | Schema-only context + LLM robustness | LLM only sees schema + 5 samples; tested in `docs/prompt_injection_report.md` | No systematic defense. Relies on LLM behavior. See "Residual risks". |
| T9 | Leaked `GROQ_API_KEY` from generated code | AST + sandbox | Sandbox environment does not have the key (only the host process does) | None known. |
| T10 | Deadlock via subprocess/zombie | Sandbox | Timeout enforced at host subprocess level | Container may leak if the host process is killed mid-run. Cleanup is best-effort. |
| T11 | Writing files to persist between runs | Sandbox | `--read-only` filesystem; only `/tmp` (tmpfs) is writable, and it's discarded on exit | None known. |
| T12 | Exfiltrating data via stderr/stdout | Sandbox | Output is captured, size-limited, and only shown to the requesting user | User can already see their own data; not a leak. |
| T13 | Docker-in-Docker escape (sandbox calling Docker) | Sandbox | Sandbox image does not include the docker CLI or socket | None if configured correctly. |
| T14 | Prompt injection to reveal the system prompt | LLM behavior | No hard defense; system prompt is not a secret | Accepted — system prompt is not confidential. |

## Layers of defense (defense in depth)
[User CSV] ──► [Schema extractor] ──► (only schema + samples reach the LLM)
│
▼
[LLM]
│
▼
[Generated code string]
│
▼
┌──────────────────────┐
│ Layer 1: AST check │ ← fast, cheap, blocks obvious
└──────────────────────┘
│
▼
┌──────────────────────┐
│ Layer 2: Exec/exec │ ← restricted builtins
│ OR Docker sandbox │ ← no network, resource limits
└──────────────────────┘

text

No single layer is sufficient. An attacker must defeat all applicable layers.

## Residual risks (honest limitations)

1. **No systematic prompt-injection defense.** The system relies on the LLM's
   own robustness plus downstream filters. A more sophisticated injection
   (multi-turn, unicode-homoglyph, encoded) might succeed. Only 3 vectors
   tested in `docs/prompt_injection_report.md`.

2. **AST checks are bypassable.** Object introspection (`__subclasses__`) can
   reach blocked modules from any live object. This is the documented reason
   the Docker sandbox is the real boundary, not the AST check.

3. **Docker-in-Docker is not tested.** The sandbox assumes a reachable Docker
   daemon. In a hosted environment (Docker-in-Docker or a compromised host),
   isolation properties depend on the container runtime's correctness.

4. **No authentication or rate limiting yet.** Phase 4 addresses this.
   Currently, any user of the Streamlit UI can query any uploaded CSV.

5. **Sandbox image drift.** If `datalens-sandbox:latest` is not rebuilt after
   a change, the sandbox behavior may diverge from the code path.

6. **Denial of service via LLM cost.** A malicious user can spam questions to
   exhaust the Groq quota. Rate limiting is on the roadmap (Phase 4).

7. **CSV parsing is trusted.** The `pd.read_csv` call runs inside the sandbox
   on the raw file, but the initial schema extraction uses the host's pandas.
   If a malicious CSV triggers a pandas bug, the host process could be affected.
   Mitigation: file size cap, future work on hardened parsing.

## Out of scope (deliberately)

- **Model extraction** — the LLM is a hosted API; extraction is the provider's problem.
- **Timing side channels** — not applicable at this abstraction level.
- **Physical access** — assume the deployment host is not physically compromised.

## What we explicitly do NOT claim

- That the sandbox is unbreakable.
- That AST checks provide security (they provide fast feedback only).
- That prompt injection is solved.
- That the system is ready for open public deployment.

## Reproduce the assessment

```bash
# Static checks
uv run pytest -v tests/test_ast_check.py

# Sandbox (requires Docker)
docker build -t datalens-sandbox:latest -f sandbox/Dockerfile sandbox/
uv run pytest -v tests/test_sandbox.py

# Prompt injection (requires real Groq key)
uv run pytest -v -m slow tests/test_prompt_injection.py


References
docs/prompt_injection_report.md — attacker payloads and observed behavior.

src/datalens/ast_check.py — Layer 1.

src/datalens/sandbox.py and sandbox/runner.py — Layer 2.

src/datalens/executor.py — dispatch logic.