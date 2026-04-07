# Ragas evaluation

ragprobe includes quantitative RAG quality evaluation using [Ragas](https://github.com/explodinggradients/ragas). Ragas provides metrics beyond pass/fail adversarial tests:

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Is the answer supported by the retrieved context? |
| Answer relevance | Does the answer address the original question? |
| Context precision | Of chunks retrieved, how many were useful? |
| Context recall | Did retrieval find all needed chunks? |

## How Ragas evaluation works

1. Run promptfoo tests to get query/answer/context pairs
2. Ragas evaluates each pair using a judge LLM
3. Scores are stored in the `probe_results` table in Postgres
4. Scores are surfaced in ragwatch and ragdeck

## probe_results table

```sql
CREATE TABLE probe_results (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    contexts JSONB NOT NULL,
    faithfulness FLOAT,
    answer_relevance FLOAT,
    context_precision FLOAT,
    context_recall FLOAT,
    ragas_score FLOAT,
    evaluated_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Running Ragas evaluation

```bash
# Against ragpipe (non-agentic)
RAGPROBE_TARGET_URL=http://localhost:8090 python scripts/run_ragas_eval.py --store --target ragpipe-v1

# Against ragorchestrator (agentic with CRAG)
RAGPROBE_TARGET_URL=http://localhost:8095 python scripts/run_ragas_eval.py --store --target ragorchestrator-crag

# Against ragorchestrator (agentic full loop)
RAGPROBE_TARGET_URL=http://localhost:8095 python scripts/run_ragas_eval.py --store --target ragorchestrator-full
```

## Comparing targets

```bash
# Compare agentic vs non-agentic
python scripts/compare_targets.py --baseline ragpipe-v1 --target ragorchestrator-crag

# With regression threshold (ignore deltas < 0.05)
python scripts/compare_targets.py --baseline baseline --target crag-v1 --threshold 0.05

# JSON output for automation
python scripts/compare_targets.py --baseline ragpipe-v1 --target ragorchestrator-crag --json
```

Exit code 1 if regressions detected. Per-route breakdown flags which specific routes regressed. Use this to prove agentic loop improves quality.

## Key files

| File | Purpose |
|------|---------|
| `ragas_eval.py` | Main Ragas evaluation logic |
| `ragas_metrics.py` | Ragas metric wrappers |
| `scripts/run_ragas_eval.py` | CLI script to run full pipeline |
| `scripts/compare_targets.py` | Compare scores between two targets |
| `ragas/corpus.yaml` | Test corpus for Ragas evaluation |
| `tests/test_compare_targets.py` | Unit tests for comparison logic |
