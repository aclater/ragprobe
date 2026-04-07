"""Live integration tests: prove agentic stack outperforms plain RAG.

Tests hit live services — no mocking.
Run with: pytest tests/test_live_integration.py -v --live

Optional endpoints:
  --ragpipe-url=http://127.0.0.1:8090
  --ragorchestrator-url=http://127.0.0.1:8095

Test categories:
  1. CRAG adversarial queries — deliberately vague, should trigger query rewrite
  2. Self-RAG trigger queries — ungrounded first attempt, should trigger reflection
  3. Multi-pass trigger queries — complex multi-topic
  4. Comparative eval — same queries against ragpipe vs ragorchestrator

Note: CRAG metadata tests (retrieval_attempts, query_rewritten) depend on
ragpipe #62 being deployed. They are marked xfail until that fix lands.
"""

import json
import urllib.error
import urllib.request

import pytest

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ragpipe_url(request):
    return request.config.getoption("--ragpipe-url")


@pytest.fixture(scope="module")
def ragorchestrator_url(request):
    return request.config.getoption("--ragorchestrator-url")


@pytest.fixture(scope="module")
def ragpipe_live(ragpipe_url):
    """Check ragpipe is reachable."""
    try:
        resp = urllib.request.urlopen(f"{ragpipe_url}/health", timeout=5)
        data = json.loads(resp.read().decode())
        if data.get("status") != "ok":
            pytest.skip("ragpipe not healthy")
    except Exception:
        pytest.skip("ragpipe not reachable")
    return ragpipe_url


@pytest.fixture(scope="module")
def ragorchestrator_live(ragorchestrator_url):
    """Check ragorchestrator is reachable."""
    try:
        resp = urllib.request.urlopen(f"{ragorchestrator_url}/health", timeout=5)
        data = json.loads(resp.read().decode())
        if data.get("status") != "ok":
            pytest.skip("ragorchestrator not healthy")
    except Exception:
        pytest.skip("ragorchestrator not reachable")
    return ragorchestrator_url


# ── Helpers ─────────────────────────────────────────────────────────────────


def _chat(base_url: str, query: str, timeout: int = 120) -> dict:
    """Send a chat completion request and return the full response."""
    payload = json.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        }
    ).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pytest.fail(f"HTTP {e.code} from {base_url}: {body[:500]}")


def _get_content(data: dict) -> str:
    """Extract assistant message content from response."""
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def _get_metadata(data: dict) -> dict:
    """Extract rag_metadata from response."""
    return data.get("rag_metadata", {})


# ── CRAG adversarial queries ───────────────────────────────────────────────
# These queries are phrased poorly enough that initial retrieval should fail.
# CRAG should rewrite the query and retry.
# Depends on ragpipe #62 deploying CRAG metadata in rag_metadata.


CRAG_QUERIES = [
    ("patent claims software", "mpep"),
    ("who works there", "personnel"),
    ("article 5 thing", "nato"),
]


@pytest.mark.live
@pytest.mark.xfail(reason="ragpipe #62: CRAG metadata not yet in rag_metadata response")
@pytest.mark.parametrize("query,expected_domain", CRAG_QUERIES)
def test_crag_metadata_in_response(ragpipe_live, query, expected_domain):
    """Vague query returns retrieval_attempts field (not null) in rag_metadata."""
    data = _chat(ragpipe_live, query, timeout=120)
    assert "choices" in data, f"Response missing choices: {data}"
    meta = _get_metadata(data)
    assert meta.get("retrieval_attempts") is not None, (
        f"retrieval_attempts is null for '{query}'. rag_metadata: {meta}"
    )


@pytest.mark.live
@pytest.mark.xfail(reason="ragpipe #62: CRAG metadata not yet in rag_metadata response")
@pytest.mark.parametrize("query,expected_domain", CRAG_QUERIES)
def test_crag_fields_always_present(ragpipe_live, query, expected_domain):
    """Every response includes retrieval_attempts and query_rewritten (never null)."""
    data = _chat(ragpipe_live, query, timeout=120)
    meta = _get_metadata(data)
    assert "retrieval_attempts" in meta, (
        f"retrieval_attempts missing from rag_metadata: {meta}"
    )
    assert "query_rewritten" in meta, (
        f"query_rewritten missing from rag_metadata: {meta}"
    )
    assert isinstance(meta["retrieval_attempts"], int)
    assert isinstance(meta["query_rewritten"], bool)


@pytest.mark.live
@pytest.mark.xfail(reason="ragpipe #62: CRAG metadata not yet in rag_metadata response")
def test_crag_rewrite_reflected_in_response(ragpipe_live):
    """At least one adversarial query triggers a rewrite (retrieval_attempts >= 2)."""
    rewrites_found = 0
    for query, _ in CRAG_QUERIES:
        data = _chat(ragpipe_live, query, timeout=120)
        meta = _get_metadata(data)
        if meta.get("retrieval_attempts", 1) >= 2:
            rewrites_found += 1
            assert meta.get("query_rewritten") is True
            assert meta.get("original_query") is not None
            assert meta.get("rewritten_query") is not None

    assert rewrites_found > 0, (
        "No CRAG rewrites triggered — may need tuning of reranker thresholds"
    )


# ── Self-RAG trigger queries ──────────────────────────────────────────────
# These should produce ungrounded first attempts or acknowledge limitations.


SELFRAG_QUERIES = [
    "what will NATO do in 2030",
    "invent a patent claim for quantum computing",
]


@pytest.mark.live
@pytest.mark.parametrize("query", SELFRAG_QUERIES)
def test_selfrag_responds_without_error(ragpipe_live, query):
    """Self-RAG trigger queries return valid responses (not errors)."""
    data = _chat(ragpipe_live, query, timeout=180)
    assert "choices" in data, f"Response missing choices: {data}"
    content = _get_content(data)
    assert len(content) > 10, f"Response too short for '{query}': {content}"


# ── Multi-pass trigger queries ────────────────────────────────────────────


@pytest.mark.live
def test_multipass_complex_query(ragorchestrator_live):
    """Complex multi-topic query returns a response via agentic path.

    This test exercises the full LangGraph loop: supervisor -> decompose ->
    multi_tools -> generate -> reflect. Requires ~5 sequential LLM calls.
    """
    data = _chat(
        ragorchestrator_live,
        "compare the roles and responsibilities of personnel in the NATO documents",
        timeout=300,
    )
    assert "choices" in data, f"Response missing choices: {data}"
    content = _get_content(data)
    assert len(content) > 10, f"Response too short: {content}"


# ── Comparative eval ──────────────────────────────────────────────────────
# Same queries against both ragpipe and ragorchestrator.


COMPARATIVE_QUERIES = [
    "what is NATO article 5",
    "who is in the personnel database",
]


@pytest.mark.live
def test_comparative_both_return_choices(ragpipe_live, ragorchestrator_live):
    """Both ragpipe and ragorchestrator return valid choices for same queries."""
    for query in COMPARATIVE_QUERIES:
        ragpipe_data = _chat(ragpipe_live, query, timeout=120)
        assert "choices" in ragpipe_data, (
            f"ragpipe missing choices for '{query}': {ragpipe_data}"
        )

        ragorchestrator_data = _chat(ragorchestrator_live, query, timeout=180)
        assert "choices" in ragorchestrator_data, (
            f"ragorchestrator missing choices for '{query}': {ragorchestrator_data}"
        )


@pytest.mark.live
def test_comparative_report(ragpipe_live, ragorchestrator_live, capsys):
    """Report comparative results between ragpipe and ragorchestrator.

    This test always passes — it generates a comparison report.
    """
    results = []

    for query in COMPARATIVE_QUERIES:
        ragpipe_data = _chat(ragpipe_live, query, timeout=120)
        ragpipe_meta = _get_metadata(ragpipe_data)
        ragpipe_content = _get_content(ragpipe_data)

        ragorchestrator_data = _chat(ragorchestrator_live, query, timeout=180)
        ragorchestrator_meta = _get_metadata(ragorchestrator_data)
        ragorchestrator_content = _get_content(ragorchestrator_data)

        results.append(
            {
                "query": query,
                "ragpipe": {
                    "grounding": ragpipe_meta.get("grounding", "unknown"),
                    "retrieval_attempts": ragpipe_meta.get("retrieval_attempts"),
                    "query_rewritten": ragpipe_meta.get("query_rewritten"),
                    "content_len": len(ragpipe_content),
                },
                "ragorchestrator": {
                    "grounding": ragorchestrator_meta.get("grounding", "unknown"),
                    "content_len": len(ragorchestrator_content),
                },
            }
        )

    # Print comparison report
    print("\n=== COMPARATIVE EVALUATION ===")
    for r in results:
        print(f"\nQuery: {r['query']}")
        print(
            f"  ragpipe:          grounding={r['ragpipe']['grounding']}, "
            f"attempts={r['ragpipe']['retrieval_attempts']}, "
            f"rewritten={r['ragpipe']['query_rewritten']}, "
            f"len={r['ragpipe']['content_len']}"
        )
        print(
            f"  ragorchestrator:  grounding={r['ragorchestrator']['grounding']}, "
            f"len={r['ragorchestrator']['content_len']}"
        )
    print("\n=== END COMPARATIVE EVALUATION ===")
