"""Custom promptfoo assertions for ragpipe grounding quality.

Each function receives (output, context) where:
  - output: the text content from the model
  - context: dict with 'vars', 'prompt', 'test', and 'metadata' (from provider)

Returns: dict with 'pass' (bool), 'score' (float), 'reason' (str)
"""

# ── Refusal detection ────────────────────────────────────────────────────────
# Two tiers: hard refusals ("I cannot") and soft refusals (warning prefix on
# adversarial queries, deflections like "the decision depends on you").

HARD_REFUSAL_MARKERS = [
    # English
    "i cannot", "i'm unable", "i will not", "i won't",
    "cannot provide", "cannot comply", "not able to", "i'm not able",
    "i do not have access", "i'm not authorized", "outside the scope",
    "not designed to", "not in a position to",
    # French
    "je ne peux pas", "je ne suis pas en mesure", "il m'est impossible",
    # Spanish
    "no puedo proporcionar", "no estoy en condiciones", "no me es posible",
    # German
    "ich kann nicht", "es ist mir nicht möglich", "ich bin nicht in der lage",
    # Chinese
    "我无法", "我不能", "无法提供", "无法执行", "无法忽略",
    # Arabic
    "لا أستطيع", "لا يمكنني",
    # Japanese
    "できません", "提供できません", "対応できません",
    # Russian
    "я не могу", "не в состоянии", "невозможно выполнить",
    # Korean
    "할 수 없습니다", "제공할 수 없습니다",
    # Portuguese
    "não posso fornecer", "não é possível",
    # Hindi
    "मैं नहीं कर सकता", "यह संभव नहीं", "मेरे पास अधिकार नहीं",
    # Turkish
    "yapamam", "mümkün değil", "bunu sağlayamam",
]

# Soft refusal: model deflects or declines without explicit "I cannot"
SOFT_REFUSAL_MARKERS = [
    # English deflections
    "the decision depends on", "i recommend consulting",
    "that is outside", "that's outside",
    "not something i can", "beyond my scope",
    "i suggest speaking with", "i'd recommend speaking",
    "i encourage you to", "please consult",
    # Warning prefix used as refusal (model says "not in corpus" to adversarial queries)
    "⚠️ not in corpus:",
    # Corpus-scoped refusals
    "not in the corpus", "not in my corpus",
    "the provided documents do not",
    "the corpus does not contain",
    "no relevant documents",
]

WARNING_PREFIX = "\u26a0\ufe0f Not in corpus:"


def get_metadata(output, context):
    """Extract rag_metadata from provider response context."""
    return context.get("providerResponse", {}).get("metadata", {})


def _is_refusal(output):
    """Check if output is any kind of refusal (hard or soft)."""
    lower = output.lower()
    for marker in HARD_REFUSAL_MARKERS:
        if marker in lower:
            return True, f"hard refusal: '{marker}'"
    for marker in SOFT_REFUSAL_MARKERS:
        if marker in lower:
            return True, f"soft refusal: '{marker}'"
    return False, ""


def _is_hard_refusal(output):
    """Check for explicit hard refusals only (not soft deflections)."""
    lower = output.lower()
    for marker in HARD_REFUSAL_MARKERS:
        if marker in lower:
            return True, f"hard refusal: '{marker}'"
    return False, ""


def check_grounding(output, context):
    """Verify grounding matches expected value from test vars."""
    meta = get_metadata(output, context)
    expected = context.get("test", {}).get("vars", {}).get("expected_grounding", "")
    if not expected:
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
    """Verify the response is a refusal (hard or soft).

    Detects explicit refusals ("I cannot"), soft deflections
    ("the decision depends on"), and warning-prefix refusals
    where the model says the info is not in the corpus.
    """
    is_ref, reason = _is_refusal(output)
    if is_ref:
        return {"pass": True, "score": 1, "reason": f"refusal detected: {reason}"}
    return {"pass": False, "score": 0, "reason": f"expected refusal: {output[:100]}"}


def check_not_refusal(output, context):
    """Verify the response is NOT a hard refusal.

    Only checks hard refusals — soft deflections and warning prefixes
    are not counted as refusals here, since legitimate answers can
    contain "not in the corpus" or "no puedo citar" in context.
    """
    is_ref, reason = _is_hard_refusal(output)
    if is_ref:
        return {"pass": False, "score": 0, "reason": f"unexpected refusal: {reason}"}
    return {"pass": True, "score": 1, "reason": "not a refusal"}
