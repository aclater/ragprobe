"""Tests for scripts/compare_targets.py comparison logic."""

import sqlite3
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))
import compare_targets


@pytest.fixture
def sqlite_db(tmp_path):
    """Create a temporary SQLite database with probe_results."""
    db_path = tmp_path / "ragprobe.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE probe_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_run_id TEXT NOT NULL,
            eval_run_at TEXT NOT NULL,
            target TEXT NOT NULL,
            ragpipe_version TEXT,
            model TEXT,
            question TEXT NOT NULL,
            ground_truth TEXT,
            answer TEXT NOT NULL,
            context_chunks TEXT NOT NULL,
            faithfulness REAL,
            answer_relevance REAL,
            context_precision REAL,
            context_recall REAL,
            routing TEXT
        )
    """)
    conn.commit()
    conn.close()
    return db_path


def _insert_results(db_path, target, scores_by_route):
    """Insert test results into the database.

    scores_by_route: dict of route -> list of (f, ar, cp, cr) tuples
    """
    conn = sqlite3.connect(db_path)
    run_id = str(uuid.uuid4())
    run_at = datetime.now(UTC).isoformat()

    for route, scores_list in scores_by_route.items():
        for i, (f, ar, cp, cr) in enumerate(scores_list):
            conn.execute(
                """INSERT INTO probe_results (
                    eval_run_id, eval_run_at, target, question, answer,
                    context_chunks, faithfulness, answer_relevance,
                    context_precision, context_recall, routing
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, run_at, target,
                    f"test question {route} {i}", "test answer",
                    "[]", f, ar, cp, cr, route,
                ),
            )
    conn.commit()
    conn.close()


class TestBuildScores:
    def test_avg_helper(self):
        assert compare_targets._avg([0.5, 0.7, 0.9]) == pytest.approx(0.7)
        assert compare_targets._avg([None, 0.8, None]) == pytest.approx(0.8)
        assert compare_targets._avg([None, None]) is None
        assert compare_targets._avg([]) is None

    def test_delta_str_positive(self):
        assert compare_targets._delta_str(0.5, 0.8) == "+0.300"

    def test_delta_str_negative(self):
        assert compare_targets._delta_str(0.8, 0.5) == "-0.300"

    def test_delta_str_none(self):
        assert compare_targets._delta_str(None, 0.5) == "N/A"
        assert compare_targets._delta_str(0.5, None) == "N/A"

    def test_is_regression_true(self):
        assert compare_targets._is_regression(0.8, 0.5, 0.0) is True

    def test_is_regression_false_when_improved(self):
        assert compare_targets._is_regression(0.5, 0.8, 0.0) is False

    def test_is_regression_within_threshold(self):
        assert compare_targets._is_regression(0.8, 0.75, 0.1) is False

    def test_is_regression_none_values(self):
        assert compare_targets._is_regression(None, 0.5, 0.0) is False
        assert compare_targets._is_regression(0.5, None, 0.0) is False


class TestQuerySQLite:
    def test_query_empty_target(self, sqlite_db):
        with patch.object(compare_targets, "SQLITE_DB", sqlite_db):
            scores = compare_targets._query_sqlite("nonexistent")
            assert scores.n == 0
            assert scores.metrics == {}
            assert scores.last_run is None

    def test_query_single_route(self, sqlite_db):
        _insert_results(sqlite_db, "baseline", {
            "personnel": [(0.9, 0.8, 0.7, 0.6)],
        })
        with patch.object(compare_targets, "SQLITE_DB", sqlite_db):
            scores = compare_targets._query_sqlite("baseline")
            assert scores.n == 1
            assert scores.metrics["faithfulness"] == pytest.approx(0.9)
            assert scores.metrics["answer_relevance"] == pytest.approx(0.8)
            assert "personnel" in scores.by_route

    def test_query_multiple_routes(self, sqlite_db):
        _insert_results(sqlite_db, "baseline", {
            "personnel": [(0.9, 0.8, 0.7, 0.6), (0.7, 0.6, 0.5, 0.4)],
            "lookup": [(0.3, 0.4, 0.5, 0.2)],
        })
        with patch.object(compare_targets, "SQLITE_DB", sqlite_db):
            scores = compare_targets._query_sqlite("baseline")
            assert scores.n == 3
            assert scores.metrics["faithfulness"] == pytest.approx((0.9 + 0.7 + 0.3) / 3)
            assert "personnel" in scores.by_route
            assert "lookup" in scores.by_route
            assert scores.by_route["personnel"]["faithfulness"] == pytest.approx(0.8)
            assert scores.by_route["lookup"]["faithfulness"] == pytest.approx(0.3)


class TestPrintComparison:
    def test_no_regressions(self, capsys):
        baseline = compare_targets.TargetScores(
            target="baseline", n=5, last_run="2026-01-01",
            metrics={"faithfulness": 0.7, "answer_relevance": 0.8,
                     "context_precision": 0.7, "context_recall": 0.25},
            by_route={},
        )
        candidate = compare_targets.TargetScores(
            target="crag-v1", n=5, last_run="2026-01-02",
            metrics={"faithfulness": 0.8, "answer_relevance": 0.85,
                     "context_precision": 0.75, "context_recall": 0.3},
            by_route={},
        )
        regressions = compare_targets.print_comparison(baseline, candidate, 0.0)
        assert len(regressions) == 0
        output = capsys.readouterr().out
        assert "No regressions detected" in output

    def test_regression_detected(self, capsys):
        baseline = compare_targets.TargetScores(
            target="baseline", n=5, last_run="2026-01-01",
            metrics={"faithfulness": 0.9, "answer_relevance": 0.8,
                     "context_precision": 0.7, "context_recall": 0.5},
            by_route={},
        )
        candidate = compare_targets.TargetScores(
            target="crag-v1", n=5, last_run="2026-01-02",
            metrics={"faithfulness": 0.5, "answer_relevance": 0.85,
                     "context_precision": 0.75, "context_recall": 0.3},
            by_route={},
        )
        regressions = compare_targets.print_comparison(baseline, candidate, 0.0)
        assert len(regressions) == 2  # faithfulness and context_recall
        output = capsys.readouterr().out
        assert "REGRESSIONS DETECTED" in output
        assert "faithfulness" in output.lower()

    def test_threshold_suppresses_small_regressions(self, capsys):
        baseline = compare_targets.TargetScores(
            target="baseline", n=5, last_run="2026-01-01",
            metrics={"faithfulness": 0.75, "answer_relevance": 0.8,
                     "context_precision": 0.7, "context_recall": 0.25},
            by_route={},
        )
        candidate = compare_targets.TargetScores(
            target="crag-v1", n=5, last_run="2026-01-02",
            metrics={"faithfulness": 0.72, "answer_relevance": 0.85,
                     "context_precision": 0.75, "context_recall": 0.25},
            by_route={},
        )
        regressions = compare_targets.print_comparison(baseline, candidate, 0.05)
        assert len(regressions) == 0

    def test_per_route_regressions(self, capsys):
        baseline = compare_targets.TargetScores(
            target="baseline", n=5, last_run="2026-01-01",
            metrics={"faithfulness": 0.7, "answer_relevance": 0.8,
                     "context_precision": 0.7, "context_recall": 0.25},
            by_route={"lookup": {"faithfulness": 0.9, "answer_relevance": 0.8,
                                  "context_precision": 0.7, "context_recall": 0.25}},
        )
        candidate = compare_targets.TargetScores(
            target="crag-v1", n=5, last_run="2026-01-02",
            metrics={"faithfulness": 0.8, "answer_relevance": 0.85,
                     "context_precision": 0.75, "context_recall": 0.3},
            by_route={"lookup": {"faithfulness": 0.4, "answer_relevance": 0.8,
                                  "context_precision": 0.7, "context_recall": 0.25}},
        )
        regressions = compare_targets.print_comparison(baseline, candidate, 0.0)
        assert any("Route lookup" in r and "faithfulness" in r for r in regressions)


class TestEndToEnd:
    def test_full_comparison_pipeline(self, sqlite_db):
        """End-to-end: insert baseline + candidate, query both, compare."""
        _insert_results(sqlite_db, "ragpipe-v1", {
            "personnel": [(0.967, 0.9, 0.85, 0.7)],
            "lookup": [(0.333, 0.6, 0.5, 0.2)],
            "general": [(0.7, 0.8, None, None)],
        })
        _insert_results(sqlite_db, "ragorchestrator-crag", {
            "personnel": [(0.95, 0.9, 0.85, 0.7)],
            "lookup": [(0.65, 0.7, 0.6, 0.35)],
            "general": [(0.7, 0.8, None, None)],
        })

        with patch.object(compare_targets, "SQLITE_DB", sqlite_db):
            baseline = compare_targets.query_target("ragpipe-v1")
            candidate = compare_targets.query_target("ragorchestrator-crag")

        assert baseline.n == 3
        assert candidate.n == 3

        # MPEP/lookup should show improvement
        assert candidate.by_route["lookup"]["faithfulness"] > baseline.by_route["lookup"]["faithfulness"]

        # Personnel slight regression is expected
        regressions = compare_targets.print_comparison(baseline, candidate, 0.05)
        # Personnel faithfulness dropped 0.967->0.95 which is within 0.05 threshold
        assert not any("personnel" in r.lower() and "faithfulness" in r for r in regressions)
