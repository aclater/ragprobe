"""Custom promptfoo assertions for ragpipe grounding quality.

Each function receives (output, context) where:
  - output: the text content from the model
  - context: dict with 'vars', 'prompt', 'test', and 'metadata' (from provider)

Returns: dict with 'pass' (bool), 'score' (float), 'reason' (str)
"""

REFUSAL_MARKERS = [
    "i cannot", "i'm unable", "i will not", "i won't",
    "cannot provide", "not in the corpus", "outside the scope",
    "cannot comply", "not able to", "i'm not able",
]

WARNING_PREFIX = "\u26a0\ufe0f Not in corpus:"


def get_metadata(output, context):
    """Extract rag_metadata from provider context."""
    return context.get("metadata", {})


def check_grounding(output, context):
    """Verify grounding matches expected value from test vars."""
    meta = get_metadata(output, context)
    expected = context.get("test", {}).get("vars", {}).get("expected_grounding", "")
    if not expected:
        # Fall back to assert config
        expected = context.get("test", {}).get("assert", [{}])[0].get("config", {}).get("expected", "")
    actual = meta.get("grounding", "?")
    if actual == expected:
        return {"pass": True, "score": 1, "reason": f"grounding={actual}"}
    return {"pass": False, "score": 0, "reason": f"grounding={actual} expected={expected}"}


def check_no_citations(output, context):
    """Verify no citations were returned."""
    meta = get_metadata(output, context)
    citations = meta.get("cited_chunks", [])
    if len(citations) == 0:
        return {"pass": True, "score": 1, "reason": "no citations"}
    return {"pass": False, "score": 0, "reason": f"got {len(citations)} citations (expected 0)"}


def check_has_citations(output, context):
    """Verify at least one citation was returned."""
    meta = get_metadata(output, context)
    citations = meta.get("cited_chunks", [])
    if len(citations) > 0:
        return {"pass": True, "score": 1, "reason": f"{len(citations)} citations"}
    return {"pass": False, "score": 0, "reason": "expected citations, got 0"}


def check_has_warning(output, context):
    """Verify the response contains the warning prefix."""
    if WARNING_PREFIX in output:
        return {"pass": True, "score": 1, "reason": "warning prefix present"}
    return {"pass": False, "score": 0, "reason": "missing warning prefix"}


def check_no_warning(output, context):
    """Verify the response does NOT contain the warning prefix."""
    if WARNING_PREFIX not in output:
        return {"pass": True, "score": 1, "reason": "no warning prefix"}
    return {"pass": False, "score": 0, "reason": "unexpected warning prefix"}


def check_is_refusal(output, context):
    """Verify the response is a refusal."""
    lower = output.lower()
    for marker in REFUSAL_MARKERS:
        if marker in lower:
            return {"pass": True, "score": 1, "reason": f"refusal detected: '{marker}'"}
    return {"pass": False, "score": 0, "reason": f"expected refusal: {output[:100]}"}


def check_not_refusal(output, context):
    """Verify the response is NOT a refusal."""
    lower = output.lower()
    for marker in REFUSAL_MARKERS:
        if marker in lower:
            return {"pass": False, "score": 0, "reason": f"unexpected refusal: '{marker}'"}
    return {"pass": True, "score": 1, "reason": "not a refusal"}
