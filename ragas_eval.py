"""Ragas evaluation for ragprobe.

Runs quantitative RAG quality metrics against ragpipe using a separate judge model.
Complementary to promptfoo's adversarial/structural tests — this measures quality.

Usage:
    python ragas_eval.py --target http://localhost:8090 --token <admin-token>
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EvalPair:
    question: str
    ground_truth: Optional[str] = None
    routing: Optional[str] = None


@dataclass
class EvalResult:
    eval_run_id: str
    eval_run_at: str
    target: str
    ragpipe_version: Optional[str]
    model: Optional[str]
    question: str
    ground_truth: Optional[str]
    answer: str
    context_chunks: list[dict]
    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    routing: Optional[str]


def load_eval_corpus(path: Path) -> list[EvalPair]:
    """Load eval corpus from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    pairs = []
    for item in data.get("eval_pairs", []):
        pairs.append(
            EvalPair(
                question=item["question"],
                ground_truth=item.get("ground_truth"),
                routing=item.get("routing"),
            )
        )
    return pairs


def call_ragpipe(
    url: str,
    question: str,
    token: str,
    model: str = "qwen3.5",
    temperature: float = 0,
) -> tuple[str, list[dict]]:
    """Call ragpipe and extract response + cited_chunks.

    Returns:
        Tuple of (answer_text, cited_chunks list)
    """
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "temperature": temperature,
        }
    ).encode()

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers=headers,
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    rag_metadata = data.get("rag_metadata", {})
    cited_chunks = rag_metadata.get("cited_chunks", [])

    return content, cited_chunks


def run_eval(
    target_url: str,
    token: str,
    corpus: list[EvalPair],
    judge_config: dict,
    target_label: str = "default",
    ragpipe_version: Optional[str] = None,
    model: Optional[str] = None,
    store_fn=None,
) -> list[EvalResult]:
    """Run Ragas evaluation against a ragpipe target.

    Args:
        target_url: ragpipe base URL (e.g. http://localhost:8090)
        token: Admin token for ragpipe
        corpus: List of eval pairs (question + optional ground_truth)
        judge_config: Dict with 'url' and 'model' for judge LLM
        target_label: Human-readable label for this target
        ragpipe_version: Git SHA of ragpipe under test
        model: Model name used for evaluation
        store_fn: Optional callback to store results (receives list[EvalResult])

    Returns:
        List of EvalResult, one per eval pair
    """
    from ragas_metrics import compute_ragas_scores, JudgeConfig

    eval_run_id = str(uuid.uuid4())
    eval_run_at = datetime.now(UTC).isoformat()
    results = []

    jc = JudgeConfig(
        url=judge_config.get("url", "http://localhost:8080"),
        model=judge_config.get("model", "qwen3.5"),
        api_key=judge_config.get("api_key"),
    )

    for pair in corpus:
        answer, cited_chunks = call_ragpipe(target_url, pair.question, token)

        contexts = [chunk.get("source", "") for chunk in cited_chunks]

        scores = compute_ragas_scores(
            question=pair.question,
            answer=answer,
            contexts=contexts,
            ground_truth=pair.ground_truth,
            judge_config=jc,
        )

        result = EvalResult(
            eval_run_id=eval_run_id,
            eval_run_at=eval_run_at,
            target=target_label,
            ragpipe_version=ragpipe_version,
            model=model,
            question=pair.question,
            ground_truth=pair.ground_truth,
            answer=answer,
            context_chunks=cited_chunks,
            faithfulness=scores.faithfulness,
            answer_relevance=scores.answer_relevance,
            context_precision=scores.context_precision,
            context_recall=scores.context_recall,
            routing=pair.routing,
        )
        results.append(result)

        print(
            f"  {pair.question[:60]:<60} | "
            f"F={scores.faithfulness:.2f if scores.faithfulness else 'N/A':>5} "
            f"AR={scores.answer_relevance:.2f if scores.answer_relevance else 'N/A':>5} "
            f"CP={scores.context_precision:.2f if scores.context_precision else 'N/A':>5} "
            f"CR={scores.context_recall:.2f if scores.context_recall else 'N/A':>5}"
        )

    if store_fn:
        store_fn(results)

    return results


def print_summary(results: list[EvalResult]) -> None:
    """Print aggregate scores."""
    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    avg_f = sum(r.faithfulness for r in results if r.faithfulness is not None) / sum(
        1 for r in results if r.faithfulness is not None
    )
    avg_ar = sum(
        r.answer_relevance for r in results if r.answer_relevance is not None
    ) / sum(1 for r in results if r.answer_relevance is not None)
    avg_cp = sum(
        r.context_precision for r in results if r.context_precision is not None
    ) / sum(1 for r in results if r.context_precision is not None)
    avg_cr = sum(
        r.context_recall for r in results if r.context_recall is not None
    ) / sum(1 for r in results if r.context_recall is not None)

    print(f"\nAggregate scores ({n} eval pairs):")
    print(f"  Faithfulness:      {avg_f:.3f}")
    print(f"  Answer Relevance:  {avg_ar:.3f}")
    print(f"  Context Precision: {avg_cp:.3f}")
    print(f"  Context Recall:    {avg_cr:.3f}")
