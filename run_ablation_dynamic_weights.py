"""
Dynamic Weight Ablation — run_ablation_dynamic_weights.py
==========================================================
Compares retrieval performance with and without dynamic weight shifting
on the human-annotated ground truth (183 queries, annotations_final.json).

Metrics:  P@5, R@5, NDCG@5
Baseline: dynamic_weights=False  (static Optuna weights)
Test:     dynamic_weights=True   (query-type-aware weight shifting)

Usage (from toolsense/ directory):
    python run_ablation_dynamic_weights.py
"""

import sys
import json
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

ANNOTATIONS_PATH = root / "data" / "annotations_final.json"
QDRANT_PATH      = str(root / "data" / "qdrant")
WEIGHTS_PATH     = root / "config" / "best_weights.yaml"


# ── Metric helpers ────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(retrieved[:k])
        if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ── Load human GT ─────────────────────────────────────────────────────────────

def load_human_gt() -> list[dict]:
    data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    queries = []
    for item in data:
        relevant = item.get("_relevant_tools", [])
        if not relevant:
            continue
        query = item["data"]["query"]
        queries.append({"query": query, "relevant": set(relevant)})
    print(f"  Loaded {len(queries)} annotated queries with >=1 relevant tool")
    return queries


# ── Load weights ──────────────────────────────────────────────────────────────

def load_weights() -> dict:
    import yaml
    if WEIGHTS_PATH.exists():
        w = yaml.safe_load(WEIGHTS_PATH.read_text())
        return {
            "w_semantic":   float(w.get("w_semantic",   0.4085)),
            "w_bm25":       float(w.get("w_bm25",       0.4500)),
            "w_category":   float(w.get("w_category",   0.1415)),
            "w_popularity": float(w.get("w_popularity", 0.0000)),
        }
    return {"w_semantic": 0.4085, "w_bm25": 0.4500, "w_category": 0.1415, "w_popularity": 0.0}


# ── Evaluate one pass ─────────────────────────────────────────────────────────

def evaluate(queries: list[dict], dynamic_weights: bool, weights: dict) -> dict:
    from src.models.ranker import retrieve, _detect_query_type

    p5_scores, r5_scores, ndcg5_scores = [], [], []
    type_counts = {"short": 0, "named_entity": 0, "conversational": 0}

    for i, item in enumerate(queries):
        query   = item["query"]
        relevant = item["relevant"]

        qtype = _detect_query_type(query)
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

        results = retrieve(
            query=query,
            qdrant_path=QDRANT_PATH,
            top_k=50,
            return_k=5,
            dynamic_weights=dynamic_weights,
            **weights,
        )
        retrieved = [r["tool_name"] for r in results]

        p5_scores.append(precision_at_k(retrieved, relevant, 5))
        r5_scores.append(recall_at_k(retrieved, relevant, 5))
        ndcg5_scores.append(ndcg_at_k(retrieved, relevant, 5))

        if (i + 1) % 30 == 0:
            print(f"    {i+1}/{len(queries)} queries done…")

    return {
        "P@5":   sum(p5_scores)    / len(p5_scores),
        "R@5":   sum(r5_scores)    / len(r5_scores),
        "NDCG@5":sum(ndcg5_scores) / len(ndcg5_scores),
        "n":     len(queries),
        "type_counts": type_counts,
    }


# ── Per-query-type breakdown ──────────────────────────────────────────────────

def evaluate_by_type(queries: list[dict], dynamic_weights: bool, weights: dict) -> dict:
    from src.models.ranker import retrieve, _detect_query_type

    by_type: dict[str, list] = {"short": [], "named_entity": [], "conversational": []}

    for item in queries:
        query    = item["query"]
        relevant = item["relevant"]
        qtype    = _detect_query_type(query)

        results   = retrieve(
            query=query,
            qdrant_path=QDRANT_PATH,
            top_k=50,
            return_k=5,
            dynamic_weights=dynamic_weights,
            **weights,
        )
        retrieved = [r["tool_name"] for r in results]
        p5        = precision_at_k(retrieved, relevant, 5)
        by_type[qtype].append(p5)

    return {qt: (sum(v)/len(v) if v else 0.0, len(v)) for qt, v in by_type.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DYNAMIC WEIGHT ABLATION — Human GT (183 queries)")
    print("=" * 60)

    queries = load_human_gt()
    weights = load_weights()
    print(f"  Weights: sem={weights['w_semantic']} | "
          f"bm25={weights['w_bm25']} | cat={weights['w_category']}\n")

    # ── Baseline: static Optuna weights ───────────────────────────────────────
    print("  [1/2] Baseline (dynamic_weights=False)…")
    base = evaluate(queries, dynamic_weights=False, weights=weights)
    print(f"        P@5={base['P@5']:.4f}  R@5={base['R@5']:.4f}  NDCG@5={base['NDCG@5']:.4f}")

    # ── Test: dynamic weight shifting ─────────────────────────────────────────
    print("\n  [2/2] Dynamic weights (dynamic_weights=True)…")
    dyn = evaluate(queries, dynamic_weights=True, weights=weights)
    print(f"        P@5={dyn['P@5']:.4f}  R@5={dyn['R@5']:.4f}  NDCG@5={dyn['NDCG@5']:.4f}")

    # ── Per-query-type breakdown for dynamic mode ─────────────────────────────
    print("\n  Query type distribution:")
    for qt, cnt in dyn["type_counts"].items():
        print(f"    {qt:<16}: {cnt} queries")

    # ── Per-type P@5 comparison ───────────────────────────────────────────────
    print("\n  Per-query-type P@5 breakdown:")
    base_by_type = evaluate_by_type(queries, dynamic_weights=False, weights=weights)
    dyn_by_type  = evaluate_by_type(queries, dynamic_weights=True,  weights=weights)

    print(f"  {'Type':<16} {'Base P@5':>9} {'Dyn P@5':>9} {'Delta':>8} {'N':>4}")
    print("  " + "-" * 50)
    for qt in ("short", "named_entity", "conversational"):
        b_p5, b_n = base_by_type[qt]
        d_p5, d_n = dyn_by_type[qt]
        delta = d_p5 - b_p5
        sign  = "+" if delta >= 0 else ""
        print(f"  {qt:<16} {b_p5:>9.4f} {d_p5:>9.4f} {sign}{delta:>7.4f} {b_n:>4}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<10} {'Baseline':>10} {'Dynamic':>10} {'Delta':>10}")
    print("  " + "-" * 44)
    for metric in ("P@5", "R@5", "NDCG@5"):
        b, d = base[metric], dyn[metric]
        delta = d - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {metric:<10} {b:>10.4f} {d:>10.4f} {sign}{delta:>9.4f}")
    print("=" * 60)

    improvement = dyn["NDCG@5"] - base["NDCG@5"]
    if improvement > 0.005:
        print(f"\n  [OK] Dynamic weights improve NDCG@5 by +{improvement:.4f}")
    elif improvement < -0.005:
        print(f"\n  [WARN]  Dynamic weights hurt NDCG@5 by {improvement:.4f}")
    else:
        print(f"\n  ≈ Neutral — delta within noise ({improvement:+.4f})")


if __name__ == "__main__":
    main()
