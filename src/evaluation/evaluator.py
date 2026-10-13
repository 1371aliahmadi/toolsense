"""
Full Performance Evaluator — Phase 06
=======================================
Computes all evaluation metrics for the ToolSense recommender:

  1. Precision@K     — fraction of top-K results that are relevant
  2. Recall@K        — fraction of relevant tools retrieved in top-K
  3. NDCG@K          — normalised discounted cumulative gain (rank-aware)
  4. Coverage        — fraction of the full tool catalogue that ever appears
                       in any top-K result across all test queries
  5. Intra-List Diversity (ILD) — average pairwise cosine distance between
                       results in the same list (higher = more diverse)

All metrics are computed at K=5 and K=10.
Results are saved as an HTML report and logged to W&B.
"""

import logging
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from src.evaluation.metrics import (
    PROXY_TEST_SET,
    is_relevant,
    precision_at_k,
    EvalResult,
)

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "base_config.yaml"
BEST_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "config" / "best_weights.yaml"


def load_weights() -> dict:
    """Load best weights from Optuna (Phase 05/05b) if available, else defaults."""
    if BEST_WEIGHTS_PATH.exists():
        with open(BEST_WEIGHTS_PATH) as f:
            w = yaml.safe_load(f)
        return {
            "w_semantic":   w.get("w_semantic",   0.80),
            "w_bm25":       w.get("w_bm25",       0.15),
            "w_category":   w.get("w_category",   0.05),
            "w_popularity": w.get("w_popularity",  0.00),
        }
    return {"w_semantic": 0.80, "w_bm25": 0.15, "w_category": 0.05, "w_popularity": 0.00}


# ── NDCG@K ────────────────────────────────────────────────────────────────────

def ndcg_at_k(results: list[dict], keywords: set[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    Relevance is binary (1 = relevant, 0 = not relevant).
    """
    top_k = results[:k]
    gains = [1.0 if is_relevant(r, keywords) else 0.0 for r in top_k]

    # DCG
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    # Ideal DCG (best possible ranking)
    n_relevant = sum(gains)
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))

    return dcg / idcg if idcg > 0 else 0.0


# ── Recall@K ──────────────────────────────────────────────────────────────────

def recall_at_k(results: list[dict], keywords: set[str], k: int) -> float:
    """
    Recall@K = (relevant tools in top-K) / (total relevant tools for this query).
    Requires keywords to be the full set of relevant tool names (ground truth mode).
    Returns 0.0 when keywords is empty (avoids division by zero).
    """
    if not keywords:
        return 0.0
    top_k = results[:k]
    hits  = sum(1 for r in top_k if is_relevant(r, keywords))
    return hits / len(keywords)


# ── Intra-List Diversity ──────────────────────────────────────────────────────

def intra_list_diversity(results: list[dict], model: SentenceTransformer) -> float:
    """
    Average pairwise cosine distance between result descriptions.
    Range [0, 1]: 0 = all identical, 1 = maximally diverse.
    """
    if len(results) < 2:
        return 0.0

    descriptions = [r.get("description", "")[:200] for r in results]
    vectors = model.encode(descriptions, normalize_embeddings=True, convert_to_numpy=True)

    distances = []
    for i, j in combinations(range(len(vectors)), 2):
        # cosine similarity → distance
        sim = float(np.dot(vectors[i], vectors[j]))
        distances.append(1.0 - sim)

    return float(np.mean(distances))


# ── Coverage ──────────────────────────────────────────────────────────────────

def catalogue_coverage(
    retrieve_fn,
    test_set: list,
    total_tools: int,
    k: int = 10,
    **retrieve_kwargs,
) -> float:
    """
    Fraction of the full catalogue that appears in at least one top-K list.
    """
    seen_tools = set()
    for query, _ in test_set:
        try:
            results = retrieve_fn(query=query, return_k=k, **retrieve_kwargs)
            for r in results:
                seen_tools.add(r["tool_name"])
        except Exception:
            pass
    return len(seen_tools) / total_tools if total_tools > 0 else 0.0


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def run_full_evaluation(
    retrieve_fn,
    qdrant_path: str,
    clean_csv_path: str,
    test_set: list = None,
    k_values: list[int] = None,
) -> dict:
    """
    Run the complete evaluation pipeline and return all metrics.
    """
    if test_set is None:
        test_set = PROXY_TEST_SET
    if k_values is None:
        k_values = [5, 10]

    weights = load_weights()
    log.info(f"Using weights: {weights}")

    # Load embedding model for diversity computation
    model_name = "all-MiniLM-L6-v2"
    log.info(f"Loading model for diversity computation: {model_name}")
    model = SentenceTransformer(model_name)

    # Load clean dataset for coverage denominator
    total_tools = len(pd.read_csv(clean_csv_path))

    retrieve_kwargs = dict(
        qdrant_path=qdrant_path,
        top_k=20,
        **weights,
    )

    # ── Per-query metrics ─────────────────────────────────────────────────────
    rows = []
    all_results_k5  = []
    all_results_k10 = []

    for query, keywords in test_set:
        try:
            results_k10 = retrieve_fn(query=query, return_k=10, **retrieve_kwargs)
            results_k5  = results_k10[:5]
        except Exception as e:
            log.warning(f"Retrieval failed for '{query}': {e}")
            results_k5 = results_k10 = []

        p5   = precision_at_k(results_k5,  keywords, 5)
        p10  = precision_at_k(results_k10, keywords, 10)
        r5   = recall_at_k(results_k5,  keywords, 5)
        r10  = recall_at_k(results_k10, keywords, 10)
        n5   = ndcg_at_k(results_k5,  keywords, 5)
        n10  = ndcg_at_k(results_k10, keywords, 10)
        ild5 = intra_list_diversity(results_k5,  model)
        ild10= intra_list_diversity(results_k10, model)

        top_tools = [r["tool_name"] for r in results_k5]

        rows.append({
            "query":       query,
            "keywords":    str(keywords),
            "p@5":         round(p5,   4),
            "p@10":        round(p10,  4),
            "r@5":         round(r5,   4),
            "r@10":        round(r10,  4),
            "ndcg@5":      round(n5,   4),
            "ndcg@10":     round(n10,  4),
            "ild@5":       round(ild5, 4),
            "ild@10":      round(ild10,4),
            "top_5_tools": ", ".join(top_tools),
        })

        all_results_k5.extend(results_k5)
        all_results_k10.extend(results_k10)

    df = pd.DataFrame(rows)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    metrics = {
        "mean_precision_at_5":  round(df["p@5"].mean(),    4),
        "mean_precision_at_10": round(df["p@10"].mean(),   4),
        "mean_recall_at_5":     round(df["r@5"].mean(),    4),
        "mean_recall_at_10":    round(df["r@10"].mean(),   4),
        "mean_ndcg_at_5":       round(df["ndcg@5"].mean(), 4),
        "mean_ndcg_at_10":      round(df["ndcg@10"].mean(),4),
        "mean_ild_at_5":        round(df["ild@5"].mean(),  4),
        "mean_ild_at_10":       round(df["ild@10"].mean(), 4),
    }

    # Coverage
    metrics["coverage_at_10"] = round(
        catalogue_coverage(retrieve_fn, test_set, total_tools, k=10, **retrieve_kwargs), 4
    )

    metrics["weights"] = weights
    metrics["n_queries"] = len(test_set)
    metrics["n_tools_in_catalogue"] = total_tools
    metrics["per_query_df"] = df

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Queries evaluated:    {len(test_set)}")
    print(f"  Catalogue size:       {total_tools} tools")
    print(f"  Weights:              semantic={weights.get('w_semantic',0)} | "
          f"bm25={weights.get('w_bm25',0)} | "
          f"category={weights.get('w_category',0)} | "
          f"popularity={weights.get('w_popularity',0)}")
    print(f"\n  Precision@5:          {metrics['mean_precision_at_5']:.4f}")
    print(f"  Precision@10:         {metrics['mean_precision_at_10']:.4f}")
    print(f"  Recall@5:             {metrics['mean_recall_at_5']:.4f}")
    print(f"  Recall@10:            {metrics['mean_recall_at_10']:.4f}")
    print(f"  NDCG@5:               {metrics['mean_ndcg_at_5']:.4f}")
    print(f"  NDCG@10:              {metrics['mean_ndcg_at_10']:.4f}")
    print(f"  Intra-List Diversity@5:  {metrics['mean_ild_at_5']:.4f}")
    print(f"  Intra-List Diversity@10: {metrics['mean_ild_at_10']:.4f}")
    print(f"  Catalogue Coverage@10:   {metrics['coverage_at_10']:.4f} "
          f"({int(metrics['coverage_at_10'] * total_tools)}/{total_tools} tools seen)")
    print("=" * 60)

    return metrics


# ── HTML report ───────────────────────────────────────────────────────────────

def save_eval_report(metrics: dict, out_path: Path) -> None:
    """Save a readable HTML evaluation report."""
    df = metrics["per_query_df"]
    w  = metrics["weights"]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ToolSense Evaluation Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; color: #333; }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }}
  .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; border-left: 4px solid #3498db; }}
  .metric-card .value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
  .metric-card .label {{ color: #7f8c8d; font-size: 0.9em; }}
  .weights {{ background: #eafaf1; border-radius: 8px; padding: 16px; margin: 16px 0; border-left: 4px solid #27ae60; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
  th {{ background: #2c3e50; color: white; padding: 10px 8px; text-align: left; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  td {{ padding: 8px; border-bottom: 1px solid #ddd; vertical-align: top; }}
  .good {{ color: #27ae60; font-weight: bold; }}
  .bad  {{ color: #e74c3c; }}
</style>
</head>
<body>
<h1>ToolSense — Performance Evaluation Report</h1>
<p>Phase 06 | Queries: {metrics['n_queries']} | Catalogue: {metrics['n_tools_in_catalogue']} tools</p>

<div class="weights">
  <strong>Tuned weights (from Optuna Phase 05):</strong>
  w_semantic = {w['w_semantic']} &nbsp;|&nbsp;
  w_category = {w['w_category']} &nbsp;|&nbsp;
  w_popularity = {w['w_popularity']}
</div>

<h2>Aggregate Metrics</h2>
<div class="metrics-grid">
  <div class="metric-card"><div class="value">{metrics['mean_precision_at_5']:.3f}</div><div class="label">Precision@5</div></div>
  <div class="metric-card"><div class="value">{metrics['mean_precision_at_10']:.3f}</div><div class="label">Precision@10</div></div>
  <div class="metric-card"><div class="value">{metrics['mean_ndcg_at_5']:.3f}</div><div class="label">NDCG@5</div></div>
  <div class="metric-card"><div class="value">{metrics['mean_ndcg_at_10']:.3f}</div><div class="label">NDCG@10</div></div>
  <div class="metric-card"><div class="value">{metrics['mean_ild_at_5']:.3f}</div><div class="label">Diversity@5</div></div>
  <div class="metric-card"><div class="value">{metrics['coverage_at_10']:.1%}</div><div class="label">Catalogue Coverage@10</div></div>
</div>

<h2>Per-Query Results</h2>
<table>
  <tr>
    <th>Query</th><th>P@5</th><th>P@10</th><th>NDCG@5</th><th>ILD@5</th><th>Top 5 Tools</th>
  </tr>
"""
    for _, row in df.iterrows():
        p5_class = "good" if row["p@5"] >= 0.6 else "bad"
        html += f"""  <tr>
    <td>{row['query']}</td>
    <td class="{p5_class}">{row['p@5']:.2f}</td>
    <td>{row['p@10']:.2f}</td>
    <td>{row['ndcg@5']:.2f}</td>
    <td>{row['ild@5']:.2f}</td>
    <td style="font-size:0.8em">{row['top_5_tools']}</td>
  </tr>
"""

    html += """</table>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info(f"Evaluation report saved to {out_path}")
