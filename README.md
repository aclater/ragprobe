# ragprobe

Adversarial testing agent for [ragpipe](https://github.com/aclater/ragpipe). Probes RAG grounding quality, citation accuracy, and safety under adversarial queries using [promptfoo](https://github.com/promptfoo/promptfoo).

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
| **Total** | **~66** | |

## Quick start

```bash
# Clone and install
git clone https://github.com/aclater/ragprobe
cd ragprobe
npm install

# Configure targets
cp targets.yaml.example targets.yaml
# Edit targets.yaml with your ragpipe URLs and tokens

# Run
npx promptfoo eval

# View results in browser
npx promptfoo view
```

## Configuration

### targets.yaml (gitignored)

Define one or more ragpipe instances to test. The comparison matrix grows automatically:

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
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGPROBE_TARGETS_FILE` | `targets.yaml` | Path to targets config |

## Iterative prompt tuning

Reload system prompts on all targets and re-run eval:

```bash
bash scripts/reload-and-eval.sh
```

Or use watch mode for continuous iteration:

```bash
npx promptfoo eval --watch
```

## Output formats

```bash
npx promptfoo eval                    # CLI table
npx promptfoo eval -o results.json    # JSON
npx promptfoo eval -o results.html    # HTML report
npx promptfoo view                    # Interactive browser UI
```

## Custom assertions

Custom Python assertions in `assertions/grounding.py` check ragpipe-specific fields:

- `check_grounding` — verifies grounding matches expected (corpus/general/mixed)
- `check_no_citations` / `check_has_citations` — citation count validation
- `check_has_warning` / `check_no_warning` — warning prefix presence
- `check_is_refusal` — detects refusal markers
- `check_not_refusal` — ensures non-refusal response

## License

AGPL-3.0-or-later
