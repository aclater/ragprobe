# ragprobe

Adversarial testing and autonomous prompt tuning agent for [ragpipe](https://github.com/aclater/ragpipe). Probes RAG grounding quality, citation accuracy, and safety under adversarial queries using [promptfoo](https://github.com/promptfoo/promptfoo), then iteratively improves ragpipe's system prompt using a local coder LLM.

![Architecture](architecture.svg)

## What it tests

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

## Quick start

```bash
# Clone and install
git clone https://github.com/aclater/ragprobe
cd ragprobe
npm install

# Configure targets
cp targets.yaml.example targets.yaml
# Edit targets.yaml with your ragpipe URLs, tokens, and agent config

# Run tests
npx promptfoo eval

# View results in browser
npx promptfoo view

# Run the autonomous tuning agent
python agent.py
```

## Configuration

### targets.yaml (gitignored)

Define ragpipe instances and agent settings:

```yaml
targets:
  - label: primary-35b
    url: http://host-a:8090
    token: admin-token-a
    model: qwen3.5
  - label: secondary-9b
    url: http://host-b:8090
    token: admin-token-b
    model: qwen3.5

agent:
  brain_url: http://host-b:8080    # local coder LLM (direct, not through ragpipe)
  brain_model: qwen3.5
  max_iterations: 5
  target_pass_rate: 0.85
  prompt_file: prompts/system-prompt.txt
```

## Tuning agent

The agent runs a closed loop: test → analyze failures → generate improved prompt → reload → re-test → learn.

```bash
python agent.py                          # 5 iterations, 85% target
python agent.py --max-iterations 1       # single iteration
python agent.py --dry-run                # analyze only, don't modify
python agent.py --target-pass-rate 0.90  # aim for 90%
```

### How it works

1. Runs `npx promptfoo eval` and parses structured results
2. Extracts failures by category with query, reason, and response preview
3. Builds an analysis prompt with current prompt + failures + iteration history
4. Calls a local coder LLM to generate an improved system prompt
5. Writes the prompt and SCPs it to remote targets
6. Calls `POST /admin/reload-prompt` on all ragpipe instances
7. Re-runs eval to verify improvement
8. **If improved**: keeps the prompt, logs success
9. **If regressed**: reverts to previous prompt, logs failure
10. Appends iteration to `history.json` so the brain learns from past attempts

### Files

| File | Purpose |
|------|---------|
| `agent.py` | Main tuning agent loop |
| `prompts/system-prompt.txt` | Current best system prompt (committed) |
| `history.json` | Iteration history for learning (gitignored) |

## Manual testing

```bash
npx promptfoo eval                    # CLI table
npx promptfoo eval -o results.json    # JSON
npx promptfoo eval -o results.html    # HTML report
npx promptfoo view                    # Interactive browser UI
bash scripts/reload-and-eval.sh       # Reload prompts + eval
```

## Custom assertions

Custom Python assertions in `assertions/grounding.py` check ragpipe-specific fields:

- `check_grounding` — verifies grounding matches expected (corpus/general/mixed)
- `check_no_citations` / `check_has_citations` — citation count validation
- `check_has_warning` / `check_no_warning` — warning prefix presence
- `check_is_refusal` — detects refusal markers in 12 languages
- `check_not_refusal` — ensures non-refusal response

## License

AGPL-3.0-or-later
