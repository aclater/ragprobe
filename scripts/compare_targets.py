#!/usr/bin/env python3
"""Compare Ragas evaluation scores between two targets.

Queries probe_results for two target labels, computes per-metric and per-route
score deltas, prints a comparison table, and flags regressions where the
candidate target scores lower than the baseline.

Usage:
    python scripts/compare_targets.py --baseline ragpipe-v1 --target ragorchestrator-crag
    python scripts/compare_targets.py --baseline baseline --target crag-v1
    python scripts/compare_targets.py --baseline baseline --target crag-v1 --threshold 0.05

Environment:
    DOCSTORE_URL   Postgres URL (optional, falls back to SQLite)

Exit codes:
    0  No regressions detected
    1  At least one metric regressed beyond threshold
"""

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
SQLITE_DB = SCRIPT_DIR / "ragprobe.db"

# Allow running scripts without pip install -e .
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]


@dataclass
class TargetScores:
    """Aggregate scores for a target, optionally broken down by route."""

    target: str
    n: int
    metrics: dict[str, float | None]
    by_route: dict[str, dict[str, float | None]]
    last_run: str | None


def _avg(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _query_sqlite(target_label: str) -> TargetScores:
    """Query SQLite for latest eval run scores."""
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row

    # Get the latest eval_run_id for this target
    row = conn.execute(
        """SELECT eval_run_id, MAX(eval_run_at) as last_run
           FROM probe_results WHERE target = ?
           GROUP BY eval_run_id ORDER BY last_run DESC LIMIT 1""",
        (target_label,),
    ).fetchone()

    if not row:
        conn.close()
        return TargetScores(
            target=target_label, n=0, metrics={}, by_route={}, last_run=None
        )

    run_id = row["eval_run_id"]
    last_run = row["last_run"]

    rows = conn.execute(
        """SELECT question, faithfulness, answer_relevance,
                  context_precision, context_recall, routing
           FROM probe_results WHERE eval_run_id = ?""",
        (run_id,),
    ).fetchall()
    conn.close()

    return _build_scores(target_label, rows, last_run)


def _query_postgres(target_label: str) -> TargetScores:
    """Query Postgres for latest eval run scores."""
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(os.environ["DOCSTORE_URL"])
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute(
        """SELECT eval_run_id, MAX(eval_run_at) as last_run
           FROM probe_results WHERE target = %s
           GROUP BY eval_run_id ORDER BY last_run DESC LIMIT 1""",
        (target_label,),
    )
    row = cursor.fetchone()

    if not row:
        conn.close()
        return TargetScores(
            target=target_label, n=0, metrics={}, by_route={}, last_run=None
        )

    run_id = row["eval_run_id"]
    last_run = str(row["last_run"])

    cursor.execute(
        """SELECT question, faithfulness, answer_relevance,
                  context_precision, context_recall, routing
           FROM probe_results WHERE eval_run_id = %s""",
        (run_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    return _build_scores(target_label, rows, last_run)


def _build_scores(target_label: str, rows: list, last_run: str | None) -> TargetScores:
    """Build TargetScores from query rows."""
    by_route: dict[str, list[dict]] = {}
    all_scores: dict[str, list[float | None]] = {m: [] for m in METRICS}

    for r in rows:
        route = r["routing"] or "unknown"
        by_route.setdefault(route, []).append(r)
        for m in METRICS:
            val = r[m]
            all_scores[m].append(float(val) if val is not None else None)

    aggregate = {m: _avg(all_scores[m]) for m in METRICS}

    route_scores = {}
    for route, route_rows in by_route.items():
        route_metrics: dict[str, list[float | None]] = {m: [] for m in METRICS}
        for r in route_rows:
            for m in METRICS:
                val = r[m]
                route_metrics[m].append(float(val) if val is not None else None)
        route_scores[route] = {m: _avg(route_metrics[m]) for m in METRICS}

    return TargetScores(
        target=target_label,
        n=len(rows),
        metrics=aggregate,
        by_route=route_scores,
        last_run=last_run,
    )


def query_target(target_label: str) -> TargetScores:
    """Query scores for a target from the appropriate storage backend."""
    if os.environ.get("DOCSTORE_URL"):
        return _query_postgres(target_label)
    if SQLITE_DB.exists():
        return _query_sqlite(target_label)
    print(f"Error: No storage backend available (no DOCSTORE_URL and no {SQLITE_DB})")
    sys.exit(1)


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _delta_str(baseline: float | None, candidate: float | None) -> str:
    if baseline is None or candidate is None:
        return "N/A"
    d = candidate - baseline
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _is_regression(
    baseline: float | None, candidate: float | None, threshold: float
) -> bool:
    if baseline is None or candidate is None:
        return False
    return (baseline - candidate) > threshold


def print_comparison(
    baseline: TargetScores, candidate: TargetScores, threshold: float
) -> list[str]:
    """Print comparison table and return list of regression descriptions."""
    regressions: list[str] = []

    print(f"\n{'=' * 72}")
    print(f"Comparison: {baseline.target} vs {candidate.target}")
    print(f"{'=' * 72}")
    print(f"  Baseline:  {baseline.target} ({baseline.n} pairs, {baseline.last_run})")
    print(f"  Candidate: {candidate.target} ({candidate.n} pairs, {candidate.last_run})")
    print(f"  Regression threshold: {threshold}")
    print()

    # Aggregate comparison
    print("Aggregate Scores:")
    print(f"  {'Metric':<20} {'Baseline':>10} {'Candidate':>10} {'Delta':>10} {'Status':>10}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for m in METRICS:
        b = baseline.metrics.get(m)
        c = candidate.metrics.get(m)
        delta = _delta_str(b, c)
        regressed = _is_regression(b, c, threshold)
        status = "REGRESSED" if regressed else ("improved" if c and b and c > b else "")
        print(f"  {m:<20} {_fmt(b):>10} {_fmt(c):>10} {delta:>10} {status:>10}")
        if regressed:
            regressions.append(f"Aggregate {m}: {_fmt(b)} -> {_fmt(c)} ({delta})")

    # Per-route comparison
    all_routes = sorted(set(list(baseline.by_route.keys()) + list(candidate.by_route.keys())))
    if all_routes:
        print(f"\nPer-Route Breakdown:")
        for route in all_routes:
            b_route = baseline.by_route.get(route, {})
            c_route = candidate.by_route.get(route, {})
            print(f"\n  Route: {route}")
            print(f"    {'Metric':<20} {'Baseline':>10} {'Candidate':>10} {'Delta':>10} {'Status':>10}")
            print(f"    {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

            for m in METRICS:
                b = b_route.get(m)
                c = c_route.get(m)
                delta = _delta_str(b, c)
                regressed = _is_regression(b, c, threshold)
                status = "REGRESSED" if regressed else (
                    "improved" if c and b and c > b else ""
                )
                print(f"    {m:<20} {_fmt(b):>10} {_fmt(c):>10} {delta:>10} {status:>10}")
                if regressed:
                    regressions.append(
                        f"Route {route} {m}: {_fmt(b)} -> {_fmt(c)} ({delta})"
                    )

    print()

    if regressions:
        print(f"REGRESSIONS DETECTED ({len(regressions)}):")
        for r in regressions:
            print(f"  - {r}")
    else:
        print("No regressions detected.")

    print()
    return regressions


def main():
    parser = argparse.ArgumentParser(
        description="Compare Ragas scores between two evaluation targets"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline target label (e.g. ragpipe-v1, baseline)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Candidate target label (e.g. ragorchestrator-crag, crag-v1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Regression threshold — flag if candidate is worse by more than this (default: 0.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output comparison as JSON",
    )
    args = parser.parse_args()

    baseline = query_target(args.baseline)
    candidate = query_target(args.target)

    if baseline.n == 0:
        print(f"Error: No results found for baseline target '{args.baseline}'")
        sys.exit(1)

    if candidate.n == 0:
        print(f"Error: No results found for candidate target '{args.target}'")
        sys.exit(1)

    if args.output_json:
        result = {
            "baseline": {
                "target": baseline.target,
                "n": baseline.n,
                "last_run": baseline.last_run,
                "metrics": baseline.metrics,
                "by_route": baseline.by_route,
            },
            "candidate": {
                "target": candidate.target,
                "n": candidate.n,
                "last_run": candidate.last_run,
                "metrics": candidate.metrics,
                "by_route": candidate.by_route,
            },
            "regressions": [],
        }
        for m in METRICS:
            if _is_regression(
                baseline.metrics.get(m), candidate.metrics.get(m), args.threshold
            ):
                result["regressions"].append(
                    {
                        "scope": "aggregate",
                        "metric": m,
                        "baseline": baseline.metrics.get(m),
                        "candidate": candidate.metrics.get(m),
                    }
                )
        print(json.dumps(result, indent=2, default=str))
        sys.exit(1 if result["regressions"] else 0)

    regressions = print_comparison(baseline, candidate, args.threshold)
    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
