# ragprobe

Adversarial testing agent for ragpipe. Uses promptfoo to run structured adversarial queries against one or more ragpipe instances and evaluate grounding quality, citation accuracy, and safety.

## How it works

1. `targets.yaml` lists ragpipe instances (URLs + admin tokens)
2. `promptfooconfig.js` dynamically builds providers from targets
3. `ragpipe_provider.py` calls ragpipe and returns rag_metadata
4. Test YAML files define queries with assertions
5. `assertions/grounding.py` checks ragpipe-specific fields
6. promptfoo runs all tests against all targets, produces comparison matrix

## Key files

```
promptfooconfig.js       — dynamic provider list from targets.yaml
ragpipe_provider.py      — custom provider returning rag_metadata
assertions/grounding.py  — check_grounding, check_no_citations, check_is_refusal, etc.
tests/*.yaml             — 13 test files, ~66 tests across 13 categories
scripts/reload-and-eval.sh — reload prompts on all targets + eval
targets.yaml             — gitignored, real URLs and tokens
```

## Running

```bash
npm install
cp targets.yaml.example targets.yaml  # edit with real URLs
npx promptfoo eval                     # run all tests
npx promptfoo eval -o results.json     # JSON output
bash scripts/reload-and-eval.sh        # reload + eval
```
