# Evaluation

How DataLens measures its own accuracy.

## Overview

DataLens uses a **golden dataset** approach: a versioned set of
(question, expected answer) pairs with manually-verified values. Every
question also has a **reference implementation** — a pandas expression
that recomputes the expected answer from the raw CSV.

## Structure
evals/
├── fixtures/ CSV files (sales, titanic, ecommerce, census)
├── questions/ Python modules with QUESTIONS lists
│ ├── sales.py
│ ├── titanic.py
│ ├── ecommerce.py
│ └── census.py
├── golden/ Generated JSON (checked into git)
│ ├── sales.json
│ ├── titanic.json
│ ├── ecommerce.json
│ └── census.json
├── build_golden.py Generates golden JSON from questions + CSV
└── run_eval.py CLI to run the eval


src/datalens/evals/
├── schema.py Pydantic models (GoldenQuestion, GoldenDataset)
├── compare.py answers_match() — tolerance-aware comparison
├── loader.py Load + validate golden JSON
├── report.py QuestionResult, EvalSummary, EvalReport
├── aggregate.py Compute summary from results
└── runner.py EvalRunner — runs golden questions


## Running evals

**Setup** (one-time):

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY
docker build -t datalens-sandbox:latest -f sandbox/Dockerfile sandbox/
uv run python evals/build_golden.py




## Running evals

**Setup** (one-time):

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY
docker build -t datalens-sandbox:latest -f sandbox/Dockerfile sandbox/
uv run python evals/build_golden.py


# All datasets, default variant
uv run python evals/run_eval.py --all

# One dataset
uv run python evals/run_eval.py titanic

# Specific variant
uv run python evals/run_eval.py --all --variant single_shot
uv run python evals/run_eval.py --all --variant graph

# Limit + delay (for rate limits)
uv run python evals/run_eval.py titanic --limit 5 --delay 3




Scalars — relative + absolute tolerance (LLM returns 57500.0 vs expected 57500)

Strings — case-insensitive, whitespace-stripped

Lists — order-independent, canonicalized

Bools — exact

Refusals — for adversarial questions; any common refusal marker passes

Adversarial questions
Two categories in the current dataset:

Nonexistent columns — e.g. "What is the total profit from all
orders?" on a dataset with no profit column. Expected kind: refusal.

Truly ambiguous questions — none currently, but the framework
supports it via low-confidence expected values.

Adversarial questions are routed through the graph variant (planner +
refuser), not single-shot.

CI integration
The eval framework is tested in CI via tests/test_eval_smoke.py with a
mocked question runner. This verifies the infrastructure (load,
run, aggregate, save) works. Real LLM calls are made only on developer
machines.

Real eval numbers are not run in CI — they require an API key,
network access, and take several minutes.

Ablation
--variant flag supports:

auto (default) — adversarial → graph; others → single-shot

single_shot — direct schema → code → executor

graph — full LangGraph agent

See docs/ablation_results.md for measured numbers.

Known limitations
Single key, single-tenant — no per-user isolation.

Provider rate limits — free tiers exhaust quickly on ablation runs.
--delay helps; production would use a paid tier or multiple providers.

No automatic retry in CI — smoke tests use mocked runners; real
evals are opt-in.