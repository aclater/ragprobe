"""Tests for ragas_eval.py print_summary per-route breakdown."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from ragas_eval import EvalResult, print_summary


class TestPrintSummaryPerRoute:
    def test_aggregate_only_when_single_route(self, capsys):
        results = [
            EvalResult(
                eval_run_id="run1",
                eval_run_at="2026-01-01T00:00:00Z",
                target="test",
                ragpipe_version=None,
                model=None,
                question="q1",
                ground_truth="a1",
                answer="a1",
                context_chunks=[],
                faithfulness=0.8,
                answer_relevance=0.85,
                context_precision=0.7,
                context_recall=0.6,
                routing="personnel",
            ),
        ]
        print_summary(results)
        output = capsys.readouterr().out
        assert "Aggregate scores (1 eval pairs)" in output
        assert "Per-Route Breakdown" not in output

    def test_per_route_breakdown_when_multiple_routes(self, capsys):
        results = [
            EvalResult(
                eval_run_id="run1",
                eval_run_at="2026-01-01T00:00:00Z",
                target="test",
                ragpipe_version=None,
                model=None,
                question="q1",
                ground_truth="a1",
                answer="a1",
                context_chunks=[],
                faithfulness=0.9,
                answer_relevance=0.85,
                context_precision=0.7,
                context_recall=0.6,
                routing="personnel",
            ),
            EvalResult(
                eval_run_id="run1",
                eval_run_at="2026-01-01T00:00:00Z",
                target="test",
                ragpipe_version=None,
                model=None,
                question="q2",
                ground_truth="a2",
                answer="a2",
                context_chunks=[],
                faithfulness=0.3,
                answer_relevance=0.6,
                context_precision=0.5,
                context_recall=0.4,
                routing="lookup",
            ),
        ]
        print_summary(results)
        output = capsys.readouterr().out
        assert "Per-Route Breakdown:" in output
        assert "Route: lookup" in output
        assert "Route: personnel" in output
        assert "0.900" in output
        assert "0.300" in output

    def test_unknown_route_when_routing_none(self, capsys):
        results = [
            EvalResult(
                eval_run_id="run1",
                eval_run_at="2026-01-01T00:00:00Z",
                target="test",
                ragpipe_version=None,
                model=None,
                question="q1",
                ground_truth="a1",
                answer="a1",
                context_chunks=[],
                faithfulness=0.8,
                answer_relevance=0.85,
                context_precision=0.7,
                context_recall=0.6,
                routing=None,
            ),
            EvalResult(
                eval_run_id="run1",
                eval_run_at="2026-01-01T00:00:00Z",
                target="test",
                ragpipe_version=None,
                model=None,
                question="q2",
                ground_truth="a2",
                answer="a2",
                context_chunks=[],
                faithfulness=0.6,
                answer_relevance=0.75,
                context_precision=0.8,
                context_recall=0.7,
                routing="personnel",
            ),
        ]
        print_summary(results)
        output = capsys.readouterr().out
        assert "Route: unknown" in output
        assert "Route: personnel" in output

    def test_empty_results(self, capsys):
        print_summary([])
        output = capsys.readouterr().out
        assert "No results to summarize" in output
