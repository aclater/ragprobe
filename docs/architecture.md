# Architecture

Adversarial testing and autonomous prompt tuning agent for [ragpipe](https://github.com/aclater/ragpipe). Probes RAG grounding quality, citation accuracy, and safety under adversarial queries using [promptfoo](https://github.com/promptfoo/promptfoo), then iteratively improves ragpipe's system prompt using a local coder LLM.

## Test categories

| Category | Tests | What it probes |
|----------|-------|----------------|
| cross_source | 5 | Citing docs from wrong context |
| hallucination | 6 | Fabricated sections, people, programs, quotes |
| leading | 6 | False premises, fabricated prior statements |
| injection | 9 | Jailbreaks, DAN, base64, YAML frontmatter, nested tags |
| role_confusion | 4 | Developer/admin/researcher/red-team claims |
| exfiltration | 4 | System prompt, infra details, corpus dump |
| scope_creep | 4 | Topic drift, personal advice, creative writing |
| temporal | 3 | Future predictions, staleness, false versions |
| confidence | 3 | Overconfident premises, false precision |
| context_poisoning | 2 | Fabricated conversation history |
| corpus_boundary | 5 | Cite/add/delete/merge/rank documents |
| multilingual | 12 | Same attacks in FR, ES, DE, ZH, AR, JA, RU, KO, PT, HI, TR, mixed |
| boundary | 3 | Empty, unicode, vague queries |
| **Total** | **66** | |

## Tuning agent loop

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

## Custom assertions

Custom Python assertions in `assertions/grounding.py` check ragpipe-specific fields:

- `check_grounding` — verifies grounding matches expected (corpus/general/mixed)
- `check_no_citations` / `check_has_citations` — citation count validation
- `check_has_warning` / `check_no_warning` — warning prefix presence
- `check_is_refusal` — detects refusal markers in 12 languages
- `check_not_refusal` — ensures non-refusal response

## Metadata format

Provider returns `{"output": content, "metadata": rag_metadata}`.
Assertions access it via `context["providerResponse"]["metadata"]`.

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
scripts/smoke-test.sh        — stack verification in under 60 seconds
scripts/reload-and-eval.sh  — manual reload + eval
```

## Prompt constraints

Brain is instructed to keep prompts under 800 chars. Oversized prompts
(>2000 chars) are truncated in code. The prompt must include citation
format [doc_id:chunk_id] and warning prefix rules.
