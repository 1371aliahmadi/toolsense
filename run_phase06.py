"""
Phase 06 Runner -- Performance Evaluation
==========================================
Runs the full evaluation pipeline using best weights from Phase 05.

Metrics computed:
  - Precision@5, Precision@10
  - Recall@5, Recall@10
  - NDCG@5, NDCG@10
  - Intra-List Diversity (ILD) @ 5 and 10
  - Catalogue Coverage @ 10

Outputs:
  - Console summary table
  - data/eval_report_phase06.html  (full HTML report)
  - W&B run: phase06-performance-eval (metrics + per-query table)

Usage (from toolsense/ directory):
    python run_phase06.py
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

QDRANT_PATH    = str(root / "data" / "qdrant")
CLEAN_CSV_PATH = str(root / "data" / "clean_tools.csv")
REPORT_PATH    = root / "data" / "eval_report_phase06.html"


def main():
    from src.models.ranker import retrieve
    from src.evaluation.evaluator import run_full_evaluation, save_eval_report

    print("=" * 60)
    print("PHASE 06 -- Performance Evaluation")
    print("=" * 60)
    print(f"  Qdrant path:  {QDRANT_PATH}")
    print(f"  Dataset:      {CLEAN_CSV_PATH}")
    print(f"  Report:       {REPORT_PATH}")
    print()

    # ── Run evaluation ────────────────────────────────────────────────────────
    metrics = run_full_evaluation(
        retrieve_fn=retrieve,
        qdrant_path=QDRANT_PATH,
        clean_csv_path=CLEAN_CSV_PATH,
    )

    # ── Save HTML report ──────────────────────────────────────────────────────
    save_eval_report(metrics, REPORT_PATH)
    print(f"\nHTML report saved: {REPORT_PATH}")

    # ── Log to W&B ────────────────────────────────────────────────────────────
    wandb_key = os.getenv("WANDB_API_KEY", "").strip()
    if wandb_key:
        try:
            import wandb
            import pandas as pd

            wandb.login(key=wandb_key)
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "toolsense"),
                name="phase06-performance-eval",
                tags=["evaluation", "metrics", "phase06"],
                config={
                    "n_queries":        metrics["n_queries"],
                    "catalogue_size":   metrics["n_tools_in_catalogue"],
                    "w_semantic":       metrics["weights"]["w_semantic"],
                    "w_category":       metrics["weights"]["w_category"],
                    "w_popularity":     metrics["weights"]["w_popularity"],
                },
            )

            # Log aggregate scalars
            wandb.log({
                "eval/precision_at_5":   metrics["mean_precision_at_5"],
                "eval/precision_at_10":  metrics["mean_precision_at_10"],
                "eval/ndcg_at_5":        metrics["mean_ndcg_at_5"],
                "eval/ndcg_at_10":       metrics["mean_ndcg_at_10"],
                "eval/ild_at_5":         metrics["mean_ild_at_5"],
                "eval/ild_at_10":        metrics["mean_ild_at_10"],
                "eval/coverage_at_10":   metrics["coverage_at_10"],
            })

            # Log per-query table
            df = metrics["per_query_df"]
            wandb_table = wandb.Table(dataframe=df)
            wandb.log({"eval/per_query_results": wandb_table})

            # Log HTML report as an artifact
            artifact = wandb.Artifact("eval_report_phase06", type="report")
            artifact.add_file(str(REPORT_PATH))
            run.log_artifact(artifact)

            wandb.finish()
            print("Results logged to W&B (project: toolsense, run: phase06-performance-eval)")
        except Exception as e:
            print(f"W&B logging failed: {e}")
    else:
        print("(W&B not configured -- skipping logging)")

    # ── Per-query table (console) ─────────────────────────────────────────────
    df = metrics["per_query_df"]
    print("\nPER-QUERY RESULTS:")
    print("-" * 80)
    print(f"{'Query':<45} {'P@5':>5} {'P@10':>5} {'NDCG@5':>7} {'ILD@5':>6}")
    print("-" * 80)
    for _, row in df.iterrows():
        q = row["query"][:43]
        print(f"{q:<45} {row['p@5']:>5.2f} {row['p@10']:>5.2f} "
              f"{row['ndcg@5']:>7.4f} {row['ild@5']:>6.4f}")
    print("-" * 80)

    # ── Final summary ─────────────────────────────────────────────────────────
    w = metrics["weights"]
    cov_n = int(metrics["coverage_at_10"] * metrics["n_tools_in_catalogue"])

    print("\n" + "=" * 60)
    print("PHASE 06 COMPLETE")
    print("=" * 60)
    print(f"  Weights: semantic={w['w_semantic']} | "
          f"category={w['w_category']} | popularity={w['w_popularity']}")
    print(f"\n  Precision@5:              {metrics['mean_precision_at_5']:.4f}")
    print(f"  Precision@10:             {metrics['mean_precision_at_10']:.4f}")
    print(f"  (Recall: computed in Phase 08 via LLM annotation)")
    print(f"  NDCG@5:                   {metrics['mean_ndcg_at_5']:.4f}")
    print(f"  NDCG@10:                  {metrics['mean_ndcg_at_10']:.4f}")
    print(f"  Intra-List Diversity@5:   {metrics['mean_ild_at_5']:.4f}")
    print(f"  Intra-List Diversity@10:  {metrics['mean_ild_at_10']:.4f}")
    print(f"  Catalogue Coverage@10:    {metrics['coverage_at_10']:.4f} "
          f"({cov_n}/{metrics['n_tools_in_catalogue']} tools seen)")
    print(f"\n  HTML report -> {REPORT_PATH.name}")
    print("=" * 60)
    print("Next step: Phase 07 -- Perturbation Analysis")


if __name__ == "__main__":
    main()
