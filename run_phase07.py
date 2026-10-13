"""
Phase 07 Runner -- Perturbation Analysis
=========================================
Tests how robust the recommender is against noisy queries.

Perturbation types tested:
  1. typo           -- character-level noise (swaps, drops, replacements)
  2. truncation     -- keep only first 40% of query words
  3. wrong_category -- append a misleading industry keyword
  4. synonym        -- replace task verbs/nouns with weaker synonyms

For each type we compare Precision@5 and NDCG@5 against the clean baseline
(Phase 06: P@5=0.85, NDCG@5=0.9437).

Outputs:
  - Console table: baseline vs. each perturbation type
  - data/perturbation_report.html
  - W&B run: phase07-perturbation-analysis

Usage (from toolsense/ directory):
    python run_phase07.py
"""

import sys
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

QDRANT_PATH = str(root / "data" / "qdrant")

# Phase 06 baseline — will be overridden by live clean-query run in main()
BASELINE_P5    = 0.8500
BASELINE_NDCG5 = 0.9437


def run_perturbation_eval(retrieve_fn, weights: dict) -> dict:
    """
    For each perturbation type, run all 20 queries perturbed and compute P@5 / NDCG@5.
    Returns dict: perturbation_type -> {p5, ndcg5, delta_p5, delta_ndcg5, per_query}
    """
    from src.evaluation.metrics import PROXY_TEST_SET, precision_at_k, is_relevant
    from src.evaluation.evaluator import ndcg_at_k
    from src.evaluation.perturbation import PERTURBATION_TYPES, apply_perturbation

    retrieve_kwargs = dict(
        qdrant_path=QDRANT_PATH,
        top_k=20,
        return_k=5,
        **weights,
    )

    results = {}

    for ptype in PERTURBATION_TYPES:
        p5_scores   = []
        ndcg5_scores = []
        per_query_rows = []

        for original_query, keywords in PROXY_TEST_SET:
            perturbed = apply_perturbation(original_query, ptype)
            try:
                hits = retrieve_fn(query=perturbed, **retrieve_kwargs)
            except Exception as e:
                log.warning(f"Retrieval failed: {e}")
                hits = []

            p5   = precision_at_k(hits, keywords, 5)
            ndcg = ndcg_at_k(hits, keywords, 5)
            p5_scores.append(p5)
            ndcg5_scores.append(ndcg)

            per_query_rows.append({
                "original_query": original_query,
                "perturbed_query": perturbed,
                "p@5": round(p5, 4),
                "ndcg@5": round(ndcg, 4),
            })

        mean_p5   = round(sum(p5_scores)    / len(p5_scores),    4)
        mean_ndcg = round(sum(ndcg5_scores) / len(ndcg5_scores), 4)

        results[ptype] = {
            "mean_p5":      mean_p5,
            "mean_ndcg5":   mean_ndcg,
            "delta_p5":     round(mean_p5   - BASELINE_P5,    4),
            "delta_ndcg5":  round(mean_ndcg - BASELINE_NDCG5, 4),
            "per_query":    per_query_rows,
        }

    return results


def save_perturbation_report(results: dict, weights: dict, out_path: Path) -> None:
    """Save HTML perturbation report."""

    def delta_cell(val: float) -> str:
        color = "#27ae60" if val >= -0.05 else "#e67e22" if val >= -0.15 else "#e74c3c"
        sign  = "+" if val >= 0 else ""
        return f'<td style="color:{color};font-weight:bold">{sign}{val:.4f}</td>'

    rows_html = ""
    for ptype, r in results.items():
        rows_html += f"""<tr>
  <td><strong>{ptype}</strong></td>
  <td>{r['mean_p5']:.4f}</td>
  {delta_cell(r['delta_p5'])}
  <td>{r['mean_ndcg5']:.4f}</td>
  {delta_cell(r['delta_ndcg5'])}
</tr>"""

    # Per-query detail sections
    detail_html = ""
    for ptype, r in results.items():
        rows = "".join(
            f"<tr><td>{row['original_query']}</td>"
            f"<td>{row['perturbed_query']}</td>"
            f"<td>{row['p@5']:.2f}</td>"
            f"<td>{row['ndcg@5']:.4f}</td></tr>"
            for row in r["per_query"]
        )
        detail_html += f"""
<h3>Perturbation: {ptype}</h3>
<table>
  <tr><th>Original Query</th><th>Perturbed Query</th><th>P@5</th><th>NDCG@5</th></tr>
  {rows}
</table>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ToolSense Perturbation Analysis</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; color: #333; }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; }}
  h3 {{ color: #555; margin-top: 28px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }}
  th {{ background: #2c3e50; color: white; padding: 10px 8px; text-align: left; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
  .baseline {{ background: #eafaf1; border-left: 4px solid #27ae60; padding: 12px; margin: 16px 0; border-radius: 6px; }}
</style>
</head>
<body>
<h1>ToolSense -- Perturbation Analysis Report</h1>
<p>Phase 07 | 4 perturbation types x 20 queries</p>

<div class="baseline">
  <strong>Baseline (Phase 06, clean queries):</strong>
  Precision@5 = {BASELINE_P5} &nbsp;|&nbsp; NDCG@5 = {BASELINE_NDCG5}<br>
  Weights: semantic={weights['w_semantic']} | category={weights['w_category']} | popularity={weights['w_popularity']}
</div>

<h2>Summary: Robustness vs. Baseline</h2>
<table>
  <tr>
    <th>Perturbation Type</th>
    <th>P@5</th><th>Delta P@5</th>
    <th>NDCG@5</th><th>Delta NDCG@5</th>
  </tr>
  <tr style="background:#eafaf1">
    <td><strong>clean (baseline)</strong></td>
    <td>{BASELINE_P5}</td><td>--</td>
    <td>{BASELINE_NDCG5}</td><td>--</td>
  </tr>
  {rows_html}
</table>

<h2>Per-Query Detail</h2>
{detail_html}
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info(f"Perturbation report saved: {out_path}")


def main():
    global BASELINE_P5, BASELINE_NDCG5
    import yaml
    from src.models.ranker import retrieve
    from src.evaluation.metrics import PROXY_TEST_SET, precision_at_k
    from src.evaluation.evaluator import ndcg_at_k

    print("=" * 60)
    print("PHASE 07 -- Perturbation Analysis")
    print("=" * 60)

    # Load best weights
    weights_path = root / "config" / "best_weights.yaml"
    with open(weights_path) as f:
        w = yaml.safe_load(f)
    weights = {
        "w_semantic":   w["w_semantic"],
        "w_bm25":       w["w_bm25"],
        "w_category":   w["w_category"],
        "w_popularity": w["w_popularity"],
    }
    print(f"  Weights: semantic={weights['w_semantic']} | bm25={weights['w_bm25']} | "
          f"category={weights['w_category']}")
    print()

    # ── Compute live clean baseline with current system ───────────────────────
    retrieve_kwargs_base = dict(qdrant_path=QDRANT_PATH, top_k=20, return_k=5, **weights)
    clean_p5_scores, clean_ndcg_scores = [], []
    for original_query, keywords in PROXY_TEST_SET:
        hits = retrieve(query=original_query, **retrieve_kwargs_base)
        clean_p5_scores.append(precision_at_k(hits, keywords, 5))
        clean_ndcg_scores.append(ndcg_at_k(hits, keywords, 5))
    BASELINE_P5    = round(sum(clean_p5_scores)   / len(clean_p5_scores),   4)
    BASELINE_NDCG5 = round(sum(clean_ndcg_scores) / len(clean_ndcg_scores), 4)
    print(f"  Live clean baseline: P@5={BASELINE_P5} | NDCG@5={BASELINE_NDCG5}")
    print()

    # ── Run perturbations ─────────────────────────────────────────────────────
    results = run_perturbation_eval(retrieve, weights)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"{'Perturbation':<20} {'P@5':>6} {'dP@5':>8} {'NDCG@5':>8} {'dNDCG':>8}")
    print("-" * 55)
    print(f"{'clean (baseline)':<20} {BASELINE_P5:>6.4f} {'--':>8} {BASELINE_NDCG5:>8.4f} {'--':>8}")
    for ptype, r in results.items():
        sign_p5   = "+" if r["delta_p5"]   >= 0 else ""
        sign_ndcg = "+" if r["delta_ndcg5"] >= 0 else ""
        print(f"{ptype:<20} {r['mean_p5']:>6.4f} "
              f"{sign_p5}{r['delta_p5']:>7.4f} "
              f"{r['mean_ndcg5']:>8.4f} "
              f"{sign_ndcg}{r['delta_ndcg5']:>7.4f}")
    print("-" * 55)

    # ── Save HTML report ──────────────────────────────────────────────────────
    report_path = root / "data" / "perturbation_report.html"
    save_perturbation_report(results, weights, report_path)
    print(f"\nHTML report saved: {report_path}")

    # ── Log to W&B ────────────────────────────────────────────────────────────
    wandb_key = os.getenv("WANDB_API_KEY", "").strip()
    if wandb_key:
        try:
            import wandb
            import pandas as pd

            wandb.login(key=wandb_key)
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "toolsense"),
                name="phase07-perturbation-analysis",
                tags=["perturbation", "robustness", "phase07"],
                config={**weights, "baseline_p5": BASELINE_P5, "baseline_ndcg5": BASELINE_NDCG5},
            )

            # Log aggregate scalars per perturbation type
            for ptype, r in results.items():
                wandb.log({
                    f"perturb/{ptype}/p5":         r["mean_p5"],
                    f"perturb/{ptype}/ndcg5":      r["mean_ndcg5"],
                    f"perturb/{ptype}/delta_p5":   r["delta_p5"],
                    f"perturb/{ptype}/delta_ndcg5":r["delta_ndcg5"],
                })

            # Summary table
            summary_rows = []
            for ptype, r in results.items():
                summary_rows.append({
                    "perturbation": ptype,
                    "mean_p5":     r["mean_p5"],
                    "delta_p5":    r["delta_p5"],
                    "mean_ndcg5":  r["mean_ndcg5"],
                    "delta_ndcg5": r["delta_ndcg5"],
                })
            df_summary = pd.DataFrame(summary_rows)
            wandb.log({"perturb/summary": wandb.Table(dataframe=df_summary)})

            # Per-query table for each perturbation type
            for ptype, r in results.items():
                df_pq = pd.DataFrame(r["per_query"])
                wandb.log({f"perturb/{ptype}/per_query": wandb.Table(dataframe=df_pq)})

            # Upload HTML artifact
            artifact = wandb.Artifact("perturbation_report", type="report")
            artifact.add_file(str(report_path))
     
            run.log_artifact(artifact)

            wandb.finish()
            print("Results logged to W&B (run: phase07-perturbation-analysis)")
        except Exception as e:
            print(f"W&B logging failed: {e}")

    # -- Final assessment
    print("\n" + "=" * 60)
    print("PHASE 07 COMPLETE -- Robustness Assessment")
    print("=" * 60)
    for ptype, r in results.items():
        drop_p5 = abs(r["delta_p5"])
        if drop_p5 <= 0.05:
            verdict = "Robust   (drop <= 5%)"
        elif drop_p5 <= 0.15:
            verdict = "Moderate (drop 5-15%)"
        else:
            verdict = "Sensitive (drop > 15%)"
        print(f"  {ptype:<20} P@5={r['mean_p5']:.4f}  [{verdict}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
