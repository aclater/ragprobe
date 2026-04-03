"""Custom promptfoo provider for ragpipe.

Returns the full rag_metadata (grounding, cited_chunks, corpus_coverage)
in the metadata dict so custom assertions can access it.
"""

import json
import urllib.request


def call_api(prompt, options, context):
    config = options.get("config", {})
    base_url = config.get("baseUrl", "http://localhost:8090")
    model = config.get("model", "qwen3.5")
    token = config.get("token", "")

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
    except Exception as e:
        return {"output": f"[ERROR: {e}]", "metadata": {}}

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    metadata = data.get("rag_metadata", {})

    return {
        "output": content,
        "metadata": metadata,
    }
