"""Ragas metric wrappers for ragprobe.

Wraps ragas library metrics with the judge model configured via environment.
Does not require ground truth for faithfulness, answer_relevance, or context_precision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ragas import evaluate
from ragas.dataset import Dataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
)


@dataclass
class JudgeConfig:
    url: str
    model: str
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            url=os.environ.get("RAGAS_JUDGE_URL", "http://localhost:8080"),
            model=os.environ.get("RAGAS_JUDGE_MODEL", "qwen3.5"),
            api_key=os.environ.get("RAGAS_JUDGE_API_KEY"),
        )


@dataclass
class RagasScores:
    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]


def compute_ragas_scores(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
    judge_config: Optional[JudgeConfig] = None,
) -> RagasScores:
    """Compute all applicable Ragas metrics for a single eval pair.

    Args:
        question: The input question.
        answer: The LLM's response.
        contexts: List of retrieved context strings.
        ground_truth: Ground truth answer (required for context_recall).
        judge_config: Judge model configuration.

    Returns:
        RagasScores with applicable metric values (None if computation failed).
    """
    if judge_config is None:
        judge_config = JudgeConfig.from_env()

    user_inputs = {"user_input": question}
    reference = {"reference": ground_truth} if ground_truth else {}
    response = {"response": answer}
    context_list = {"contexts": contexts}

    row = {**user_inputs, **reference, **response, **context_list}
    dataset = Dataset.from_rows([row])

    metrics = [
        Faithfulness(name="faithfulness"),
        AnswerRelevance(name="answer_relevance"),
        ContextPrecision(name="context_precision"),
    ]

    if ground_truth:
        metrics.append(ContextRecall(name="context_recall"))

    result = evaluate(
        dataset, metrics=metrics, llm=judge_config.url, embedding=judge_config.url
    )

    scores = result.scores[0]

    return RagasScores(
        faithfulness=_safe_get(scores, "faithfulness"),
        answer_relevance=_safe_get(scores, "answer_relevance"),
        context_precision=_safe_get(scores, "context_precision"),
        context_recall=_safe_get(scores, "context_recall"),
    )


def _safe_get(d: dict, key: str) -> Optional[float]:
    """Safely extract a score from ragas result dict."""
    try:
        val = d.get(key)
        if val is None:
            return None
        return float(val)
    except (TypeError, ValueError):
        return None
