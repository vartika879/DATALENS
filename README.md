# DataLens

Natural-language data analysis over CSV files — powered by an agentic AI workflow with isolated code execution, evaluation, and observability.

> Status: Phase 0 — project skeleton in progress.

## Roadmap

- [ ] Phase 0: Setup, tooling, Docker fundamentals
- [ ] Phase 1: CSV upload → schema → LLM-generated pandas → result
- [ ] Phase 2: LangGraph workflow + bounded repair loop
- [ ] Phase 3: Sandbox hardening + threat model
- [ ] Phase 4: FastAPI + persistence
- [ ] Phase 5: Golden eval dataset + ablation
- [ ] Phase 6: Observability (tracing, cost, latency)
- [ ] Phase 7: CI/CD + deploy
- [ ] Phase 8: Docs, demo, portfolio packaging

## Quickstart

```bash
git clone <this-repo>
cd datalens
uv sync
cp .env.example .env   # fill values when needed
uv run pytest