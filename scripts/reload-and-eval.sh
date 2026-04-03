#!/usr/bin/env bash
# Reload ragpipe system prompts on all targets, then run promptfoo eval.
set -euo pipefail

TARGETS_FILE="${RAGPROBE_TARGETS_FILE:-targets.yaml}"

if [[ ! -f "$TARGETS_FILE" ]]; then
    echo "ERROR: $TARGETS_FILE not found. Copy targets.yaml.example to targets.yaml and configure." >&2
    exit 1
fi

echo "Reloading system prompts..."
python3 -c "
import yaml, urllib.request, json, sys
targets = yaml.safe_load(open('$TARGETS_FILE'))['targets']
for t in targets:
    try:
        req = urllib.request.Request(
            f'{t[\"url\"]}/admin/reload-prompt',
            method='POST',
            headers={'Authorization': f'Bearer {t.get(\"token\", \"\")}'},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
            print(f'  {t[\"label\"]}: {data.get(\"status\", \"?\")} (changed={data.get(\"changed\", \"?\")})')
    except Exception as e:
        print(f'  {t[\"label\"]}: ERROR — {e}', file=sys.stderr)
"

echo ""
npx promptfoo eval "$@"
