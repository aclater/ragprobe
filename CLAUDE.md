# ragprobe

Adversarial testing and autonomous prompt tuning agent for ragpipe.

## Architecture
```
agent.py loop:
  1. npx promptfoo eval -o results.json
  2. Parse failures by category (66 tests, 13 categories)
  3. Build analysis prompt: current prompt + failures + history
  4. Call local coder LLM (Qwen3-Coder-30B-A3B) for improved prompt
  5. Write prompt + SCP to remote targets
  6. POST /admin/reload-prompt on all targets
  7. Re-run eval → compare
  8. Keep if improved, revert if regressed
  9. Append to history.json for learning
```

## Key files
```
agent.py                    — tuning loop (eval → brain → reload → verify)
ragpipe_provider.py         — custom promptfoo provider returning rag_metadata
assertions/grounding.py    — 7 assertion functions, 40+ multilingual refusal markers
promptfooconfig.js          — dynamic provider list from targets.yaml
prompts/system-prompt.txt   — current best prompt (committed)
targets.yaml                — gitignored, real URLs + tokens + agent config
history.json                — gitignored, iteration history for learning
tests/*.yaml                — 13 test files, 66 adversarial tests
scripts/reload-and-eval.sh  — manual reload + eval
```

## Test categories (66 tests total)

| Category | Count |
|----------|-------|
| cross_source | 5 |
| hallucination | 6 |
| leading | 6 |
| injection | 9 |
| role_confusion | 4 |
| exfiltration | 4 |
| scope_creep | 4 |
| temporal | 3 |
| confidence | 3 |
| context_poisoning | 2 |
| corpus_boundary | 5 |
| multilingual | 12 |
| boundary | 3 |

## Agent config (in targets.yaml)
```yaml
agent:
  brain_url: http://host:8080       # coder LLM direct (NOT through ragpipe)
  brain_model: qwen3-coder
  max_iterations: 5
  target_pass_rate: 0.85
  prompt_file: prompts/system-prompt.txt
```

## Running
```bash
python agent.py --dry-run              # analyze only
python agent.py --max-iterations 1     # single iteration
python agent.py                        # full 5-iteration run
npx promptfoo eval                     # manual test run
```

## How metadata reaches assertions
Provider returns `{"output": content, "metadata": rag_metadata}`.
Assertions access it via `context["providerResponse"]["metadata"]`.

## Prompt constraints
Brain is instructed to keep prompts under 800 chars. Oversized prompts
(>2000 chars) are truncated in code. The prompt must include citation
format [doc_id:chunk_id] and warning prefix rules.

## rag_metadata.cited_chunks format (v3)

cited_chunks entries are objects with `id`, `title`, and `source`:
```python
cited_chunks = [
    {"id": "abc-123:0", "title": "Q3 Strategy", "source": "gdrive://file.pdf"}
]
```

Assertions should extract IDs:
```python
chunk_ids = [c["id"] for c in context["providerResponse"]["metadata"]["cited_chunks"]]
```

## Ragas evaluation

ragprobe includes quantitative RAG quality evaluation using Ragas.
Metrics are stored in the `probe_results` table in Postgres.

### probe_results table
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

### Running Ragas evaluation
```bash
# Against ragpipe (non-agentic)
RAGPROBE_TARGET_URL=http://localhost:8090 python scripts/run_ragas_eval.py --store --target ragpipe-v1

# Against ragorchestrator (agentic with CRAG)
RAGPROBE_TARGET_URL=http://localhost:8095 python scripts/run_ragas_eval.py --store --target ragorchestrator-crag

# Against ragorchestrator (agentic full loop)
RAGPROBE_TARGET_URL=http://localhost:8095 python scripts/run_ragas_eval.py --store --target ragorchestrator-full
```

### Comparing targets
```bash
# Compare agentic vs non-agentic
python scripts/compare_targets.py --baseline ragpipe-v1 --target ragorchestrator-crag

# With regression threshold (ignore deltas < 0.05)
python scripts/compare_targets.py --baseline baseline --target crag-v1 --threshold 0.05

# JSON output for automation
python scripts/compare_targets.py --baseline ragpipe-v1 --target ragorchestrator-crag --json
```

Exit code 1 if regressions detected. Per-route breakdown flags which specific
routes regressed. Use this to prove agentic loop improves quality.

### Key files
```
ragas_eval.py               — Ragas evaluation logic
ragas_metrics.py            — Ragas metric wrappers
scripts/run_ragas_eval.py   — CLI script to run full pipeline
scripts/compare_targets.py  — Compare scores between two targets
ragas/corpus.yaml           — Test corpus for Ragas evaluation
tests/test_compare_targets.py — Unit tests for comparison logic
```
