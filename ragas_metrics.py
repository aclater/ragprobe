"""Ragas-equivalent metric scoring for ragprobe.

Uses LLM-as-judge via direct HTTP calls instead of the ragas library,
which is incompatible with Python 3.14 (asyncio.wait_for + sniffio).

Metrics:
- Faithfulness: Is the answer supported by the contexts?
- Answer Relevance: Does the answer address the question?
- Context Precision: Are the retrieved contexts relevant?
- Context Recall: Do the contexts cover the ground truth? (requires ground_truth)
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.request
from dataclasses import dataclass
from typing import Optional

# Force IPv4 — llama-vulkan via pasta only listens on IPv4
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, *args, **kwargs):
    return _orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)


socket.getaddrinfo = _ipv4_getaddrinfo


@dataclass
class JudgeConfig:
    url: str
    model: str
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> JudgeConfig:
        return cls(
            url=os.environ.get("RAGAS_JUDGE_URL", "http://localhost:8080"),
            model=os.environ.get("RAGAS_JUDGE_MODEL", "model.file"),
            api_key=os.environ.get("RAGAS_JUDGE_API_KEY"),
        )


@dataclass
class RagasScores:
    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]


def _call_judge(prompt: str, judge: JudgeConfig) -> str:
    """Call the judge LLM and return the response text, with retries."""
    import time

    # /nothink disables Qwen3 thinking mode so the model returns scores directly
    payload = json.dumps({
        "model": judge.model,
        "messages": [{"role": "user", "content": f"/nothink\n{prompt}"}],
        "temperature": 0,
        "max_tokens": 50,
    }).encode()

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{judge.url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            return data["choices"][0]["message"].get("content", "")
        except Exception as e:
            if attempt < 2:
                wait = 5 * (attempt + 1)
                time.sleep(wait)
            else:
                raise


def _parse_score(text: str) -> Optional[float]:
    """Extract a 0.0-1.0 score from judge response."""
    m = re.search(r"(?:score[:\s=]*)?(\d+\.?\d*)", text.lower())
    if m:
        v = float(m.group(1))
        if v > 1.0:
            v = v / 10.0 if v <= 10.0 else v / 100.0
        return max(0.0, min(1.0, v))
    return None


def _judge_faithfulness(question, answer, contexts, judge):
    if not contexts:
        return None
    ctx = "\n---\n".join(contexts[:5])
    prompt = (
        f"Rate how faithful the answer is to the provided contexts on a scale of 0.0 to 1.0.\n"
        f"1.0 = every claim in the answer is supported by the contexts\n"
        f"0.0 = the answer contradicts or fabricates beyond the contexts\n\n"
        f"Question: {question}\nContexts: {ctx[:3000]}\nAnswer: {answer[:2000]}\n\n"
        f"Respond with ONLY a number between 0.0 and 1.0. Nothing else."
    )
    return _parse_score(_call_judge(prompt, judge))


def _judge_answer_relevance(question, answer, judge):
    prompt = (
        f"Rate how relevant the answer is to the question on a scale of 0.0 to 1.0.\n"
        f"1.0 = the answer directly and completely addresses the question\n"
        f"0.0 = the answer is completely off-topic\n\n"
        f"Question: {question}\nAnswer: {answer[:2000]}\n\n"
        f"Respond with ONLY a number between 0.0 and 1.0. Nothing else."
    )
    return _parse_score(_call_judge(prompt, judge))


def _judge_context_precision(question, contexts, judge):
    ctx = "\n---\n".join(contexts[:5])
    prompt = (
        f"Rate how relevant the retrieved contexts are to the question on a scale of 0.0 to 1.0.\n"
        f"1.0 = all contexts are highly relevant to answering the question\n"
        f"0.0 = none of the contexts are relevant\n\n"
        f"Question: {question}\nContexts: {ctx[:3000]}\n\n"
        f"Respond with ONLY a number between 0.0 and 1.0. Nothing else."
    )
    return _parse_score(_call_judge(prompt, judge))


def _judge_context_recall(question, contexts, ground_truth, judge):
    ctx = "\n---\n".join(contexts[:5]) if contexts else "No contexts retrieved"
    prompt = (
        f"Rate how well the retrieved contexts cover the information needed to produce "
        f"the reference answer, on a scale of 0.0 to 1.0.\n"
        f"1.0 = the contexts contain all information needed for the reference answer\n"
        f"0.0 = the contexts contain none of the needed information\n\n"
        f"Question: {question}\nReference answer: {ground_truth}\nContexts: {ctx[:3000]}\n\n"
        f"Respond with ONLY a number between 0.0 and 1.0. Nothing else."
    )
    return _parse_score(_call_judge(prompt, judge))


def compute_ragas_scores(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
    judge_config: Optional[JudgeConfig] = None,
) -> RagasScores:
    """Compute all applicable Ragas-equivalent metrics for a single eval pair.

    Uses LLM-as-judge via direct HTTP calls (Python 3.14 compatible).
    """
    if judge_config is None:
        judge_config = JudgeConfig.from_env()

    try:
        faithfulness = _judge_faithfulness(question, answer, contexts, judge_config)
        answer_relevance = _judge_answer_relevance(question, answer, judge_config)
        context_precision = (
            _judge_context_precision(question, contexts, judge_config) if contexts else None
        )
        context_recall = (
            _judge_context_recall(question, contexts, ground_truth, judge_config)
            if ground_truth
            else None
        )
    except Exception:
        return RagasScores(
            faithfulness=None,
            answer_relevance=None,
            context_precision=None,
            context_recall=None,
        )

    return RagasScores(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        context_recall=context_recall,
    )
