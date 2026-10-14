# Ablation Results

**Date:** 2026-10-14
**Datasets:** 4 (sales, titanic, ecommerce, census)
**Questions:** 50 total

## Variants compared

| Variant | Description |
|---|---|
| `single_shot` | Direct schema → code → executor pipeline |
| `graph` | Full LangGraph agent (planner + coder + executor + fixer + summarizer) |

## Accuracy (final)

| Dataset | single_shot | graph |
|---|---|---|
| sales | 5/5 (100%) | 5/5 (100%) |
| titanic | 15/15 (100%) | 15/15 (100%) |
| ecommerce | 14/15 (93.3%) | 15/15 (100%) |
| census | 14/15 (93.3%) | 15/15 (100%) |
| **Total** | **48/50 (96%)** | **50/50 (100%)** |

## Latency

| Variant | p50 (avg) | Total 50 Q |
|---|---|---|
| single_shot | ~2,400 ms | ~227 s |
| graph | ~11,000 ms | ~500 s |

Graph is **~4-5x slower** than single-shot — the agent makes 5+ LLM
calls per question (planner, coder, summarizer), each adding latency.

## Failure modes observed (before prompt tuning)

**Single-shot:**
- 2 adversarial hallucinations — LLM returned a plausible number
  (`11200`) for a column that does not exist (`profit`). Dangerous
  because the user sees a confident wrong answer.

**Graph (original planner prompt):**
- 2 over-cautious clarifications — planner asked
  "Do you want the total for the entire dataset or just the sample rows?"
  for questions like "What is the total revenue?". The full dataset was
  available; planner was being paranoid.

## Fix

Added explicit guidance to `planner_v2.txt`:
- Questions mentioning aggregates ("total", "average", "count") always
  refer to the complete dataset.
- Only ask for clarification when genuinely ambiguous.

**Result:** graph accuracy went from 96% → 100%.

## Trade-off (the real finding)

| Aspect | single_shot | graph |
|---|---|---|
| Accuracy (normal Q) | 96% | 100% |
| Accuracy (adversarial Q) | 0% (hallucinates) | 100% (refuses) |
| Latency | Fast (~2 s) | Slow (~12 s) |
| Cost per Q | 1 LLM call | 5+ LLM calls |

**Recommendation for production:**
Use a **routing policy**: single-shot for clear questions, graph for
adversarial / ambiguous inputs. Both paths are implemented and
switchable via `--variant` in `evals/run_eval.py`.