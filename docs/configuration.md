# Configuration

## targets.yaml (gitignored)

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

## Manual testing

```bash
npx promptfoo eval                    # CLI table
npx promptfoo eval -o results.json    # JSON
npx promptfoo eval -o results.html   # HTML report
npx promptfoo view                    # Interactive browser UI
bash scripts/reload-and-eval.sh       # Reload prompts + eval
```
