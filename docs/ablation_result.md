# Ablation Results

**Date:** 2026-10-14
**Datasets:** 4 (sales, titanic, ecommerce, census)
**Questions:** 50 total

## Variants compared

| Variant | Description |
|---|---|
| `single_shot` | Direct schema → code → executor pipeline |
| `graph` | Full LangGraph agent (planner + coder + executor + fixer + summarizer) |
| `auto` | Adversarial → graph; others → single-shot |

## Accuracy

| Variant | Correct | Total | Accuracy |
|---|---|---|---|
| single_shot | 48 | 50 | 96.0% |
| graph | 48 | 50 | 96.0% |

## Latency

| Variant | p50 | Total 50 Q |
|---|---|---|
| single_shot | ~2,400 ms | ~227 s |
| graph | ~12,000 ms | ~563 s |

## Key observations

1. **Accuracy is identical (96%)** across both variants on this test set.
2. **Graph is ~5x slower** than single-shot — the agent makes 5+ LLM calls
   per question (planner, coder, summarizer), each adding latency.
3. **Failure modes differ:**
   - Single-shot: 2 adversarial hallucinations (returns a plausible
     number for a column that does not exist).
   - Graph: 2 transient failures (rate limit related).

## Trade-off

The graph variant is **safer** for adversarial inputs — the planner refuses
questions about nonexistent columns instead of hallucinating. However, this
safety comes at a 5x latency cost. For production, one would:
- Use `auto` (single-shot for normal, graph for adversarial)
- Or implement caching to amortize LLM calls
- Or pay for higher-tier LLM throughput