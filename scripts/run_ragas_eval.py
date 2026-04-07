#!/usr/bin/env python3
"""CLI for ragprobe Ragas evaluation.

Run quantitative RAG quality metrics against ragpipe targets.

Usage:
    # Against ragpipe (non-agentic baseline)
    python scripts/run_ragas_eval.py --target-url http://localhost:8090 --target baseline --store

    # Against ragorchestrator (agentic)
    python scripts/run_ragas_eval.py --target-url http://localhost:8095 --target crag-v1 --store

    # Compare two targets
    python scripts/compare_targets.py --baseline baseline --target crag-v1

    # Show latest stored scores
    python scripts/run_ragas_eval.py --latest --target baseline

    # Using targets.yaml (optional)
    python scripts/run_ragas_eval.py --store --target primary-35b

    # Custom judge model
    python scripts/run_ragas_eval.py --target-url http://localhost:8090 --judge http://lennon:8080

Environment:
    RAGPROBE_TARGET_URL     Default target URL when --target-url not given
    RAGPROBE_TARGETS_FILE   Path to targets.yaml (default: targets.yaml)
    RAGAS_JUDGE_URL         Judge LLM URL (default: http://localhost:8080)
    RAGAS_JUDGE_MODEL       Judge model name (default: model.file)
    DOCSTORE_URL            Postgres URL for storing results (optional, falls back to SQLite)
"""

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
DEFAULT_CORPUS = SCRIPT_DIR / "ragas" / "corpus.yaml"
SQLITE_DB = SCRIPT_DIR / "ragprobe.db"


def load_targets():
    """Load targets from targets.yaml."""
    import yaml

    targets_file = os.environ.get("RAGPROBE_TARGETS_FILE", SCRIPT_DIR / "targets.yaml")
    if not Path(targets_file).exists():
        return {}

    with open(targets_file) as f:
        data = yaml.safe_load(f)
    return data


def load_corpus(corpus_path: Path):
    """Load eval corpus from YAML."""
    from ragas_eval import load_eval_corpus

    return load_eval_corpus(corpus_path)


def get_storage():
    """Get storage function based on environment.

    Returns (store_fn, init_fn) tuple.
    """
    docstore_url = os.environ.get("DOCSTORE_URL")

    if docstore_url:
        return _store_postgres, lambda: None
    else:
        _init_sqlite()
        return _store_sqlite, _init_sqlite


def _init_sqlite():
    """Initialize SQLite database."""
    conn = sqlite3.connect(SQLITE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS probe_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_run_id TEXT NOT NULL,
            eval_run_at TIMESTAMPTZ NOT NULL,
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


def _store_sqlite(results: list) -> None:
    """Store results in SQLite."""
    import ragas_eval

    conn = sqlite3.connect(SQLITE_DB)
    for r in results:
        if isinstance(r, ragas_eval.EvalResult):
            r = asdict(r)
        conn.execute(
            """
            INSERT INTO probe_results (
                eval_run_id, eval_run_at, target, ragpipe_version, model,
                question, ground_truth, answer, context_chunks,
                faithfulness, answer_relevance, context_precision, context_recall, routing
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                r["eval_run_id"],
                r["eval_run_at"],
                r["target"],
                r.get("ragpipe_version"),
                r.get("model"),
                r["question"],
                r.get("ground_truth"),
                r["answer"],
                json.dumps(r["context_chunks"]),
                r.get("faithfulness"),
                r.get("answer_relevance"),
                r.get("context_precision"),
                r.get("context_recall"),
                r.get("routing"),
            ),
        )
    conn.commit()
    conn.close()


def _store_postgres(results: list) -> None:
    """Store results in Postgres (rag-suite)."""
    import psycopg2
    from ragas_eval import EvalResult

    conn = psycopg2.connect(os.environ["DOCSTORE_URL"])
    cursor = conn.cursor()
    for r in results:
        if isinstance(r, EvalResult):
            r = asdict(r)
        cursor.execute(
            """
            INSERT INTO probe_results (
                eval_run_id, eval_run_at, target, ragpipe_version, model,
                question, ground_truth, answer, context_chunks,
                faithfulness, answer_relevance, context_precision, context_recall, routing
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                r["eval_run_id"],
                r["eval_run_at"],
                r["target"],
                r.get("ragpipe_version"),
                r.get("model"),
                r["question"],
                r.get("ground_truth"),
                r["answer"],
                json.dumps(r["context_chunks"]),
                r.get("faithfulness"),
                r.get("answer_relevance"),
                r.get("context_precision"),
                r.get("context_recall"),
                r.get("routing"),
            ),
        )
    conn.commit()
    conn.close()


def run_single_target(
    target_url: str,
    token: str,
    corpus: list,
    judge_config: dict,
    target_label: str,
    store: bool,
):
    """Run evaluation against a single target."""
    from ragas_eval import run_eval, print_summary

    ragpipe_version = os.environ.get("RAGPIPE_VERSION")

    store_fn, _ = get_storage() if store else (None, None)

    results = run_eval(
        target_url=target_url,
        token=token,
        corpus=corpus,
        judge_config=judge_config,
        target_label=target_label,
        ragpipe_version=ragpipe_version,
        store_fn=store_fn,
    )

    print_summary(results)
    return results


def get_latest_scores(target_label: str) -> dict:
    """Get latest scores for a target from storage."""
    store_fn, _ = get_storage()

    if store_fn == _store_sqlite:
        conn = sqlite3.connect(SQLITE_DB)
        cursor = conn.execute(
            """
            SELECT
                AVG(faithfulness) as avg_f,
                AVG(answer_relevance) as avg_ar,
                AVG(context_precision) as avg_cp,
                AVG(context_recall) as avg_cr,
                COUNT(*) as n,
                MAX(eval_run_at) as last_run
            FROM probe_results
            WHERE target = ?
        """,
            (target_label,),
        )
        row = cursor.fetchone()
        conn.close()
        if row and row[0] is not None:
            result = {
                "faithfulness": row[0],
                "answer_relevance": row[1],
                "context_precision": row[2],
                "context_recall": row[3],
                "n": row[4],
                "last_run": row[5],
            }

            conn2 = sqlite3.connect(SQLITE_DB)
            route_cursor = conn2.execute(
                """
                SELECT
                    routing,
                    AVG(faithfulness) as avg_f,
                    AVG(answer_relevance) as avg_ar,
                    AVG(context_precision) as avg_cp,
                    AVG(context_recall) as avg_cr,
                    COUNT(*) as n
                FROM probe_results
                WHERE target = ?
                GROUP BY routing
                """,
                (target_label,),
            )
            by_route = {}
            for route_row in route_cursor.fetchall():
                routing = route_row[0] or "unknown"
                by_route[routing] = {
                    "faithfulness": route_row[1],
                    "answer_relevance": route_row[2],
                    "context_precision": route_row[3],
                    "context_recall": route_row[4],
                    "n": route_row[5],
                }
            conn2.close()
            result["by_route"] = by_route
            return result
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run Ragas evaluation against ragpipe")
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS, help="Path to eval corpus YAML"
    )
    parser.add_argument("--target", help="Target label from targets.yaml")
    parser.add_argument("--target-url", help="Direct ragpipe URL")
    parser.add_argument("--token", help="Admin token (or set RAGPIPE_ADMIN_TOKEN)")
    parser.add_argument("--judge", help="Judge LLM URL (or set RAGAS_JUDGE_URL)")
    parser.add_argument(
        "--judge-model", help="Judge model name (or set RAGAS_JUDGE_MODEL)"
    )
    parser.add_argument("--store", action="store_true", help="Store results")
    parser.add_argument(
        "--latest", action="store_true", help="Show latest scores from storage"
    )
    args = parser.parse_args()

    if args.latest:
        if not args.target:
            print("Error: --target required for --latest")
            sys.exit(1)
        scores = get_latest_scores(args.target)
        if scores:
            print(
                f"Latest scores for {args.target} ({scores['n']} pairs, {scores['last_run']}):"
            )
            print(f"  Faithfulness:      {scores['faithfulness']:.3f}")
            print(f"  Answer Relevance:  {scores['answer_relevance']:.3f}")
            print(f"  Context Precision: {scores['context_precision']:.3f}")
            print(f"  Context Recall:    {scores['context_recall']:.3f}")
            by_route = scores.get("by_route", {})
            if len(by_route) > 1:
                print("\nPer-Route Breakdown:")
                for route, rs in sorted(by_route.items()):
                    print(f"  Route: {route} ({rs['n']} pairs)")
                    print(f"    Faithfulness:      {rs['faithfulness']:.3f}")
                    print(f"    Answer Relevance:  {rs['answer_relevance']:.3f}")
                    print(f"    Context Precision: {rs['context_precision']:.3f}")
                    print(f"    Context Recall:    {rs['context_recall']:.3f}")
        else:
            print(f"No stored results for {args.target}")
        sys.exit(0)

    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} eval pairs from {args.corpus}")

    judge_config = {
        "url": args.judge or os.environ.get("RAGAS_JUDGE_URL", "http://localhost:8080"),
        "model": args.judge_model or os.environ.get("RAGAS_JUDGE_MODEL", "qwen3.5"),
    }

    target_url = args.target_url or os.environ.get("RAGPROBE_TARGET_URL")

    if target_url:
        token = args.token or os.environ.get("RAGPIPE_ADMIN_TOKEN", "")
        run_single_target(
            target_url=target_url,
            token=token,
            corpus=corpus,
            judge_config=judge_config,
            target_label=args.target or "direct",
            store=args.store,
        )
    else:
        targets = load_targets()
        if not targets:
            print("Error: No target specified. Use one of:")
            print("  python scripts/run_ragas_eval.py --target-url http://localhost:8090 --target baseline")
            print("  RAGPROBE_TARGET_URL=http://localhost:8090 python scripts/run_ragas_eval.py --target baseline")
            print("  Create targets.yaml (see targets.yaml.example)")
            sys.exit(1)

        for tgt in targets.get("targets", []):
            label = tgt.get("label", "unknown")
            if args.target and args.target != label:
                continue

            print(f"\nEvaluating target: {label} ({tgt.get('url')})")

            run_single_target(
                target_url=tgt.get("url", ""),
                token=tgt.get("token", ""),
                corpus=corpus,
                judge_config=judge_config,
                target_label=label,
                store=args.store,
            )


if __name__ == "__main__":
    main()
