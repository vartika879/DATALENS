# DataLens

> Ask questions about your CSV in plain English. DataLens generates pandas code, executes it in a restricted environment, and returns the answer — with the code and explanation shown transparently.

[![tests](https://img.shields.io/badge/tests-51%20passing-brightgreen)](#testing)
[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](#license)
[![status](https://img.shields.io/badge/status-phase%201%20of%208-orange)](#roadmap)

**Status:** Phase 1 complete — working single-shot MVP. Agentic loop, sandbox, persistence, and evaluation are on the [roadmap](#roadmap).

---

## Overview

DataLens is an **agentic AI system** for natural-language data analysis, built as a portfolio project to demonstrate production-grade AI engineering — not just LLM calls.

Instead of sending an entire CSV to an LLM (expensive, privacy-unsafe), DataLens:

1. Extracts a compact **schema** (columns, dtypes, null counts, sample rows).
2. Asks an LLM to generate **pandas code** with a structured JSON response (`code`, `explanation`, `self_reported_confidence`).
3. Executes the code in a **restricted environment** with size limits and full error capture.
4. Returns the answer to the user along with the generated code and a plain-English explanation.

Every stage is explicit, tested, and measured.

---

## Demo

> **Q:** *"in which month highest sale"*

```
Answer: 11

Total latency:       949 ms
Code execution:       71 ms
Model estimate:       0.95

Generated code:
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
    month_sales = df.groupby(df['OrderDate'].dt.month)['Sales'].sum()
    result = int(month_sales.idxmax())

Explanation:
    Finds the month with the highest total sales by converting sales to
    numeric, parsing dates, summing per month, and selecting the month
    with the maximum sum.
```

*(A demo GIF will be added after Phase 3.)*

---

## What works today

| Capability | Status | Notes |
|---|---|---|
| CSV upload + validation | ✅ | Size cap (20 MB), empty-file detection, parse-error handling |
| Schema extraction | ✅ | Columns, dtypes, null counts, 5 sample rows — no full data sent to LLM |
| LLM code generation | ✅ | Groq, JSON mode, versioned system prompt |
| Structured output | ✅ | Pydantic-validated `GeneratedCode` |
| Restricted code execution | ⚠️ | Whitelisted `__builtins__`, size limits — **not a security boundary yet** |
| Streamlit UI | ✅ | Upload, ask, view answer + code + explanation |
| Docker reproducibility | ✅ | `docker compose up ui` |
| Tests | ✅ | 51 tests, local + Docker |

**What is not built yet:** agentic retry loop, sandbox isolation, authentication, persistence, evaluation suite. See [Roadmap](#roadmap).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py)                                     │
│  CSV upload, question input, result rendering               │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  pipeline.run_pipeline                                      │
│  Orchestrates: schema → LLM → executor, measures total time │
└─────┬─────────────────────┬─────────────────────┬───────────┘
      │                     │                     │
      ▼                     ▼                     ▼
┌──────────┐        ┌──────────────┐      ┌────────────────┐
│ schema   │        │ llm          │      │ executor       │
│ extract  │        │ generate_code│      │ execute_code   │
│          │        │  (Groq, JSON)│      │ (restricted)   │
└──────────┘        └──────────────┘      └────────────────┘
```

**Key design choice:** business logic lives in `src/datalens/`, not in `app.py`. The Streamlit UI is a thin wrapper. This makes the Phase 4 FastAPI backend swap trivial.

---

## Quickstart

### Prerequisites

- **Python 3.12+** — managed by [uv](https://docs.astral.sh/uv/)
- **Docker Desktop** — for the containerized workflow
- **Groq API key** — free tier at [console.groq.com](https://console.groq.com)

### Setup

```bash
git clone https://github.com/<your-username>/datalens.git
cd datalens
cp .env.example .env
# Open .env and set: GROQ_API_KEY=gsk_...
```

### Run locally

```bash
uv sync
uv run streamlit run app.py
```

Open <http://localhost:8501>, upload a CSV, ask a question.

### Run in Docker

```bash
docker compose build
docker compose up ui
```

### Run tests and linters

```bash
# Local
uv run pytest -v
uv run ruff check .
uv run ruff format . --check

# Docker
docker compose run --rm test
docker compose run --rm lint
```

---

## Security and limitations

Security is a first-class concern in this project, and honesty about its limits is a feature.

### What is protected today

- **File validation** — size cap (20 MB), empty-file rejection, parse-error surfacing.
- **Restricted `__builtins__`** for executed code — `import`, `open`, `eval`, `exec` are blocked.
- **Result size limits** — caps on rows and repr length, so a runaway result cannot crash the UI or blow up the JSON payload.
- **Secrets management** — API keys loaded from environment variables; `.env` is gitignored and never committed.
- **Non-root user** in the Docker image.

### What is **not** yet protected

- **`exec` is not a security boundary.** Restricted builtins stop *accidental* misuse (a generated `import os`, an `open()` call), but a determined attacker can escape via object introspection. **Real isolation — Docker sandbox with no network, CPU/RAM/time limits, read-only filesystem, non-root user — is scheduled for Phase 3.**
- No authentication or rate limiting yet (Phase 4).
- No evaluation of answer correctness yet (Phase 5). **Accuracy is not claimed.**

### What we explicitly do not claim

- **LLM self-reported confidence is not a calibrated probability.** It is exposed as `self_reported_confidence` with a UI tooltip stating this, and is treated as a hint only. A derived confidence (execution success, retry count, sanity checks) is planned for Phase 2.
- **Successful execution is not proof of correctness.** The code ran without error; the answer may still be wrong.

This section will be updated as Phase 3 and Phase 5 land.

---

## Design decisions

Short rationales for the choices a reviewer is most likely to question.

**Why not send the full CSV to the LLM?**
Cost and privacy. Sending a 2 MB CSV on every question is expensive and leaks user data. The schema (columns, dtypes, null counts, 5 sample rows) is enough for the model to write correct pandas code in most cases. The full file never leaves the server.

**Why structured output (Pydantic) instead of parsing free text?**
The LLM returns JSON with `code`, `explanation`, and `self_reported_confidence`. Pydantic validates types and ranges *before* anything runs. Structured outputs make downstream logic simpler and safer.

**Why a separate `pipeline.py`?**
The Streamlit UI and the future FastAPI backend both call `run_pipeline(schema, question, df)`. The business logic does not know who is calling it. This keeps the UI thin and makes the Phase 4 backend swap a routing change, not a rewrite.

**Why `self_reported_confidence` instead of `confidence`?**
Because the model's self-estimate is not calibrated. Naming it explicitly prevents downstream misuse. A derived confidence field is planned for Phase 2.

**Why a versioned prompt file (`generate_code_v1.txt`)?**
So Phase 5 evaluation runs can compare prompt versions. Prompt changes become measurable diffs, not vibes. The file lives *inside the package* (`src/datalens/prompts/`) so it ships with the wheel and works in every environment — this was a real bug fixed during Phase 1.

**Why Docker for the dev environment?**
Reproducibility. `git clone && docker compose run --rm test` works on any machine, any OS. This matters because the whole project is designed to run untrusted code later — the same container tooling will be extended in Phase 3.

**Why `compile()` and `exec()` as separate steps?**
`compile()` raises `SyntaxError` for malformed code, while `exec()` raises runtime exceptions. Keeping them separate means the Phase 2 Fixer node can distinguish *syntactically broken* code from *semantically broken* code — the repair strategies differ.

---

## Roadmap

- [x] **Phase 0** — Project setup, Docker, tooling, tests
- [x] **Phase 1** — CSV → schema → LLM → executor → Streamlit MVP
- [ ] **Phase 2** — LangGraph agent: Planner → Coder → Executor → Fixer → Summarizer, with bounded retries
- [ ] **Phase 3** — Docker sandbox hardening: no network, resource limits, read-only FS, non-root, threat model
- [ ] **Phase 4** — FastAPI backend, SQLite persistence, file validation, auth, rate limiting
- [ ] **Phase 5** — Golden evaluation dataset, correctness metrics, ablation experiments, failure analysis
- [ ] **Phase 6** — Observability: Langfuse traces, structured logs, cost and latency monitoring
- [ ] **Phase 7** — CI/CD, deployment
- [ ] **Phase 8** — Documentation, demo video, evaluation report

---

## Evaluation

> **Not yet available.** This section will be filled in during Phase 5.

The plan is to build a golden dataset of 60–100 questions across public datasets (Titanic, an e-commerce sales dataset, an Indian dataset like GST or census) with independently verified expected answers, covering:

- Easy (single aggregation)
- Medium (groupby, filters)
- Hard (joins, window functions, time series)
- Adversarial (wrong column name, ambiguous question, unanswerable)

Metrics will include execution success rate, answer correctness (with numeric tolerance), retries per question, latency (p50/p95), and cost per query.

An ablation table will compare:

| Variant | Accuracy | Avg cost |
|---|---|---|
| Single-shot (baseline) | TBD | TBD |
| + Fixer loop | TBD | TBD |
| + Planner | TBD | TBD |
| Small model vs large | TBD | TBD |

**No accuracy numbers will be reported until they are measured on this dataset.**

---

## Project structure

```
datalens/
├── app.py                          Streamlit UI (thin wrapper)
├── Dockerfile
├── docker-compose.yml              Services: test, lint, ui
├── pyproject.toml                  uv-managed, ruff + pytest config
├── .env.example                    Env var names, no values
├── src/datalens/
│   ├── config.py                   Settings (typed, cached, .env-aware)
│   ├── schema.py                   CSV → DatasetSchema
│   ├── serialization.py            JSON-safe conversion of pandas/numpy values
│   ├── llm.py                      Groq client, JSON mode, prompt loading
│   ├── executor.py                 Restricted exec + result capture
│   ├── pipeline.py                 End-to-end orchestration
│   └── prompts/
│       └── generate_code_v1.txt    Versioned system prompt
├── tests/                          51 tests (schema, llm, executor, pipeline, serialization)
├── data/                           Local CSV uploads (gitignored)
├── prompts/                        Future: eval prompts
└── evals/                          Future: golden dataset + runner
```

---

## Testing

```bash
uv run pytest -v
```

Current coverage:

| File | Tests | What it covers |
|---|---|---|
| `test_schema.py` | 11 | Validation, schema extraction, edge cases (empty, header-only, unicode) |
| `test_llm.py` | 14 | Prompt loading, structured output parsing, error handling (all mocked) |
| `test_executor.py` | 14 | Success/failure paths, restricted builtins, result serialization |
| `test_serialization.py` | 7 | NaN, numpy scalars, Timestamps → JSON-safe |
| `test_pipeline.py` | 4 | End-to-end orchestration (LLM mocked) |
| `test_smoke.py` | 1 | Package import + version |
| **Total** | **51** | |

LLM calls are always mocked in unit tests. Real API calls happen only in the manual smoke test.

---

## Contributing

This is a personal portfolio project under active development. Issues and suggestions are welcome, but APIs may change between phases.

## License

MIT