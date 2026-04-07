# ragprobe

Adversarial testing and autonomous prompt tuning agent for [ragpipe](https://github.com/aclater/ragpipe). Probes RAG grounding quality, citation accuracy, and safety under adversarial queries using [promptfoo](https://github.com/promptfoo/promptfoo), then iteratively improves ragpipe's system prompt using a local coder LLM.

![Architecture](architecture.svg)

## Table of contents

- [Architecture](docs/architecture.md) — data flow, test categories, tuning agent loop
- [Configuration](docs/configuration.md) — targets.yaml, tuning agent usage
- [Ragas evaluation](docs/ragas-evaluation.md) — quantitative quality metrics and comparison

## Quick start

```bash
# Clone and install
git clone https://github.com/aclater/ragprobe
cd ragprobe
npm install
pip install -e .

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

## Running tests

```bash
npx promptfoo eval                    # CLI table
npx promptfoo eval -o results.json    # JSON
npx promptfoo eval -o results.html    # HTML report
npx promptfoo view                    # Interactive browser UI
bash scripts/reload-and-eval.sh      # Reload prompts + eval
```

## License

AGPL-3.0-or-later
