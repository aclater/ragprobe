#!/usr/bin/env python3
"""Phase 0 baseline evaluation — records ragpipe quality before agentic migration.

Runs each eval pair through ragpipe, hydrates context text from the Postgres
docstore, computes Ragas metrics via the local judge model, stores results
in the probe_results table, and writes BASELINE.md with aggregate scores.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --corpus tests/ragas_eval.yaml
    python scripts/run_baseline.py --dry-run  # skip Ragas, just query ragpipe

Environment:
    RAGPIPE_URL          ragpipe endpoint (default: http://localhost:8090)
    RAGPIPE_ADMIN_TOKEN  auth token
    RAGAS_JUDGE_URL      judge LLM endpoint (default: http://localhost:8080)
    RAGAS_JUDGE_MODEL    judge model name (default: model.file)
    DOCSTORE_URL         Postgres URL for storing results and hydrating chunks
"""

import argparse
import json
import os
import socket
import sys
import urllib.request
import uuid
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

# Force IPv4 — llama-vulkan container only listens on IPv4 via pasta,
# IPv6 connections to localhost get RST
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, *args, **kwargs):
    return _orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)
socket.getaddrinfo = _ipv4_getaddrinfo

SCRIPT_DIR = Path(__file__).parent.parent


@dataclass
class EvalPair:
    question: str
    ground_truth: Optional[str] = None
    routing: Optional[str] = None


@dataclass
class BaselineResult:
    eval_run_id: str
    eval_run_at: str
    target: str
    model: Optional[str]
    question: str
    ground_truth: Optional[str]
    answer: str
    context_chunks: list[dict]
    grounding: str
    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    routing: Optional[str]


def load_corpus(path: Path) -> list[EvalPair]:
    """Load eval corpus from YAML."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        EvalPair(
            question=item["question"],
            ground_truth=item.get("ground_truth"),
            routing=item.get("routing"),
        )
        for item in data.get("eval_pairs", [])
    ]


def call_ragpipe(url: str, question: str, token: str) -> dict:
    """Call ragpipe and return full response data."""
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        "stream": False,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=payload, headers=headers
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def hydrate_contexts(cited_chunks: list[dict], docstore_url: str) -> list[str]:
    """Fetch chunk text from Postgres docstore for Ragas context evaluation.

    cited_chunks entries have format: {"id": "doc_id:chunk_id", "title": ..., "source": ...}
    The chunks table PK is (doc_id, chunk_id) with text in the `text` column.
    """
    if not cited_chunks or not docstore_url:
        return []

    import psycopg2

    texts = []
    conn = psycopg2.connect(docstore_url)
    cursor = conn.cursor()
    try:
        for chunk in cited_chunks:
            chunk_id_str = chunk.get("id", "")
            if ":" not in chunk_id_str:
                continue
            doc_id, chunk_idx = chunk_id_str.rsplit(":", 1)
            try:
                chunk_idx = int(chunk_idx)
            except ValueError:
                continue
            cursor.execute(
                "SELECT text FROM chunks WHERE doc_id = %s AND chunk_id = %s",
                (doc_id, chunk_idx),
            )
            row = cursor.fetchone()
            if row:
                texts.append(row[0])
    finally:
        conn.close()

    return texts


def _call_judge(prompt: str, judge_url: str, judge_model: str) -> str:
    """Call the judge LLM and return the response text, with retries."""
    import time

    # Prefix with /nothink to disable Qwen3.5 thinking mode for judge calls.
    # Without this, the model spends all tokens on reasoning_content and
    # returns empty content.
    payload = json.dumps({
        "model": judge_model,
        "messages": [{"role": "user", "content": f"/nothink\n{prompt}"}],
        "temperature": 0,
        "max_tokens": 50,
    }).encode()

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{judge_url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            return data["choices"][0]["message"].get("content", "")
        except Exception as e:
            if attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"      judge retry {attempt + 1}/3 after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def _parse_score(text: str) -> Optional[float]:
    """Extract a 0.0-1.0 score from judge response."""
    import re
    # Look for patterns like "Score: 0.8" or "0.85" or "score=0.7"
    m = re.search(r'(?:score[:\s=]*)?(\d+\.?\d*)', text.lower())
    if m:
        v = float(m.group(1))
        if v > 1.0:
            v = v / 10.0 if v <= 10.0 else v / 100.0
        return max(0.0, min(1.0, v))
    return None


def _judge_faithfulness(question, answer, contexts, judge_url, judge_model):
    """Is the answer faithful to the provided contexts?"""
    if not contexts:
        return None
    ctx = "\n---\n".join(contexts[:5])
    prompt = f"""Rate how faithful the answer is to the provided contexts on a scale of 0.0 to 1.0.
1.0 = every claim in the answer is supported by the contexts
0.0 = the answer contradicts or fabricates beyond the contexts

Question: {question}
Contexts: {ctx[:3000]}
Answer: {answer[:2000]}

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""
    return _parse_score(_call_judge(prompt, judge_url, judge_model))


def _judge_answer_relevance(question, answer, judge_url, judge_model):
    """Is the answer relevant to the question asked?"""
    prompt = f"""Rate how relevant the answer is to the question on a scale of 0.0 to 1.0.
1.0 = the answer directly and completely addresses the question
0.0 = the answer is completely off-topic

Question: {question}
Answer: {answer[:2000]}

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""
    return _parse_score(_call_judge(prompt, judge_url, judge_model))


def _judge_context_precision(question, contexts, judge_url, judge_model):
    """Are the retrieved contexts relevant to the question?"""
    ctx = "\n---\n".join(contexts[:5])
    prompt = f"""Rate how relevant the retrieved contexts are to the question on a scale of 0.0 to 1.0.
1.0 = all contexts are highly relevant to answering the question
0.0 = none of the contexts are relevant

Question: {question}
Contexts: {ctx[:3000]}

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""
    return _parse_score(_call_judge(prompt, judge_url, judge_model))


def _judge_context_recall(question, contexts, ground_truth, judge_url, judge_model):
    """Do the contexts contain the information needed for the ground truth answer?"""
    ctx = "\n---\n".join(contexts[:5]) if contexts else "No contexts retrieved"
    prompt = f"""Rate how well the retrieved contexts cover the information needed to produce the reference answer, on a scale of 0.0 to 1.0.
1.0 = the contexts contain all information needed for the reference answer
0.0 = the contexts contain none of the needed information

Question: {question}
Reference answer: {ground_truth}
Contexts: {ctx[:3000]}

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""
    return _parse_score(_call_judge(prompt, judge_url, judge_model))


def compute_ragas(question, answer, contexts, ground_truth=None,
                  judge_url="http://localhost:8080", judge_model="model.file"):
    """Compute Ragas metrics using the judge model."""
    # Ragas library is incompatible with Python 3.14 asyncio (wait_for + sniffio).
    # Implement Ragas-equivalent LLM-as-judge scoring via direct HTTP calls.
    try:
        scores = {}

        scores["faithfulness"] = _judge_faithfulness(
            question, answer, contexts, judge_url, judge_model)

        scores["answer_relevance"] = _judge_answer_relevance(
            question, answer, judge_url, judge_model)

        if contexts:
            scores["context_precision"] = _judge_context_precision(
                question, contexts, judge_url, judge_model)
        else:
            scores["context_precision"] = None

        if ground_truth:
            scores["context_recall"] = _judge_context_recall(
                question, contexts, ground_truth, judge_url, judge_model)
        else:
            scores["context_recall"] = None

        return scores
    except Exception as e:
        print(f"    Ragas error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "faithfulness": None,
            "answer_relevance": None,
            "context_precision": None,
            "context_recall": None,
        }


def _safe_float(v):
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN check
            return None
        return f
    except (TypeError, ValueError):
        return None


def store_postgres(results: list[BaselineResult], docstore_url: str):
    """Store results in Postgres probe_results table."""
    import psycopg2

    conn = psycopg2.connect(docstore_url)
    cursor = conn.cursor()
    for r in results:
        d = asdict(r)
        cursor.execute("""
            INSERT INTO probe_results (
                eval_run_id, eval_run_at, target, model,
                question, ground_truth, answer, context_chunks,
                faithfulness, answer_relevance, context_precision,
                context_recall, routing
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            d["eval_run_id"], d["eval_run_at"], d["target"], d["model"],
            d["question"], d["ground_truth"], d["answer"],
            json.dumps(d["context_chunks"]),
            d["faithfulness"], d["answer_relevance"],
            d["context_precision"], d["context_recall"], d["routing"],
        ))
    conn.commit()
    conn.close()
    print(f"\nStored {len(results)} results in probe_results table")


def write_baseline_md(results: list[BaselineResult], path: Path, model_desc: str):
    """Write BASELINE.md with aggregate and per-route scores."""
    by_route: dict[str, list[BaselineResult]] = {}
    for r in results:
        route = r.routing or "unknown"
        by_route.setdefault(route, []).append(r)

    def avg(vals):
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else None

    def fmt(v):
        return f"{v:.3f}" if v is not None else "N/A"

    lines = [
        "# Ragpipe Quality Baseline (Phase 0)",
        "",
        f"**Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Eval pairs:** {len(results)}",
        f"**Target:** ragpipe ({model_desc} via Vulkan/gfx1151)",
        f"**Judge:** {model_desc} (same model, direct endpoint)",
        "",
        "## Aggregate Scores",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Faithfulness | {fmt(avg([r.faithfulness for r in results]))} |",
        f"| Answer Relevance | {fmt(avg([r.answer_relevance for r in results]))} |",
        f"| Context Precision | {fmt(avg([r.context_precision for r in results]))} |",
        f"| Context Recall | {fmt(avg([r.context_recall for r in results]))} |",
        "",
        "## Per-Route Scores",
        "",
        "| Route | N | Faithfulness | Answer Rel. | Context Prec. | Context Recall |",
        "|-------|---|--------------|-------------|---------------|----------------|",
    ]

    for route in ["personnel", "analysis", "lookup", "general"]:
        rr = by_route.get(route, [])
        if not rr:
            continue
        lines.append(
            f"| {route} | {len(rr)} | "
            f"{fmt(avg([r.faithfulness for r in rr]))} | "
            f"{fmt(avg([r.answer_relevance for r in rr]))} | "
            f"{fmt(avg([r.context_precision for r in rr]))} | "
            f"{fmt(avg([r.context_recall for r in rr]))} |"
        )

    lines.extend([
        "",
        "## Per-Query Results",
        "",
        "| # | Route | Question | Grounding | F | AR | CP | CR |",
        "|---|-------|----------|-----------|---|----|----|-----|",
    ])

    for i, r in enumerate(results, 1):
        q = r.question[:50] + ("..." if len(r.question) > 50 else "")
        lines.append(
            f"| {i} | {r.routing or '?'} | {q} | {r.grounding} | "
            f"{fmt(r.faithfulness)} | {fmt(r.answer_relevance)} | "
            f"{fmt(r.context_precision)} | {fmt(r.context_recall)} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- This baseline was recorded before any agentic RAG improvements (CRAG, Self-RAG, etc.)",
        "- The judge model is the same LLM as the response model — scores may be optimistic",
        "- Context Recall requires ground_truth; queries without it show N/A",
        "- General route has `rag_enabled: false` so context metrics reflect no-RAG behavior",
        "- Contexts hydrated from Postgres docstore (chunks table) using cited chunk IDs",
        "- All future agentic improvements must demonstrate improvement relative to these scores",
        "",
    ])

    path.write_text("\n".join(lines))
    print(f"\nBaseline written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0 baseline evaluation")
    parser.add_argument(
        "--corpus", type=Path,
        default=SCRIPT_DIR / "ragas" / "corpus.yaml",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Query ragpipe only, skip Ragas metrics",
    )
    args = parser.parse_args()

    ragpipe_url = os.environ.get("RAGPIPE_URL", "http://localhost:8090")
    token = os.environ.get("RAGPIPE_ADMIN_TOKEN", "change-me")
    judge_url = os.environ.get("RAGAS_JUDGE_URL", "http://localhost:8080")
    judge_model = os.environ.get("RAGAS_JUDGE_MODEL", "model.file")
    docstore_url = os.environ.get(
        "DOCSTORE_URL",
        "postgresql://litellm:litellm@localhost:5432/litellm",
    )
    model_desc = "Qwen3-32B Q4_K_M"

    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} eval pairs from {args.corpus}")
    print(f"Target: {ragpipe_url}")
    print(f"Judge: {judge_url} ({judge_model})")
    print(f"Mode: {'dry-run (no Ragas)' if args.dry_run else 'full eval'}")
    print()

    eval_run_id = str(uuid.uuid4())
    eval_run_at = datetime.now(UTC).isoformat()
    results: list[BaselineResult] = []

    import time

    for i, pair in enumerate(corpus, 1):
        print(f"[{i}/{len(corpus)}] {pair.routing or '?':>10} | {pair.question[:60]}")

        try:
            data = call_ragpipe(ragpipe_url, pair.question, token)
        except Exception as e:
            print(f"    ERROR calling ragpipe: {e}")
            continue

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        meta = data.get("rag_metadata", {})
        grounding = meta.get("grounding", "?")
        cited = meta.get("cited_chunks", [])

        print(f"    grounding={grounding}, cited={len(cited)}, answer={content[:80]}...")

        if args.dry_run:
            scores = {
                "faithfulness": None,
                "answer_relevance": None,
                "context_precision": None,
                "context_recall": None,
            }
        else:
            # Hydrate context text from Postgres docstore
            contexts = hydrate_contexts(cited, docstore_url)
            print(f"    hydrated {len(contexts)} context chunks from docstore")

            scores = compute_ragas(
                pair.question, content, contexts,
                ground_truth=pair.ground_truth,
                judge_url=judge_url, judge_model=judge_model,
            )
            print(
                f"    F={scores['faithfulness']}, "
                f"AR={scores['answer_relevance']}, "
                f"CP={scores['context_precision']}, "
                f"CR={scores['context_recall']}"
            )

        # Brief pause between evals to avoid overwhelming the single model
        if i < len(corpus):
            time.sleep(2)

        results.append(BaselineResult(
            eval_run_id=eval_run_id,
            eval_run_at=eval_run_at,
            target="baseline",
            model=model_desc,
            question=pair.question,
            ground_truth=pair.ground_truth,
            answer=content,
            context_chunks=cited,
            grounding=grounding,
            faithfulness=scores["faithfulness"],
            answer_relevance=scores["answer_relevance"],
            context_precision=scores["context_precision"],
            context_recall=scores["context_recall"],
            routing=pair.routing,
        ))

    # Write BASELINE.md
    baseline_path = SCRIPT_DIR / "BASELINE.md"
    write_baseline_md(results, baseline_path, model_desc)

    # Store in Postgres
    try:
        store_postgres(results, docstore_url)
    except Exception as e:
        print(f"Warning: Failed to store in Postgres: {e}")


if __name__ == "__main__":
    main()
