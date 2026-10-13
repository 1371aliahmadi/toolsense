"""
Phase 05b — Optuna Re-tuning with Real Ground Truth
=====================================================
Re-runs Optuna weight optimisation using the Claude-annotated ground truth
from Phase 08 instead of the keyword proxy set.

This produces weights optimised against actual relevance labels, not keyword
substring matching. Expected to improve Phase 08 Precision@K.

Usage (from toolsense/ directory):
    python run_phase05b.py

Outputs:
  - Best weights saved to config/best_weights.yaml (overwrites Phase 05 weights)
  - W&B run: phase05b-optuna-gt-tuning
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
logging.getLogger("src").setLevel(logging.WARNING)

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

import yaml
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

QDRANT_PATH   = str(root / "data" / "qdrant")
GT_PATH       = root / "data" / "ground_truth.json"
N_TRIALS      = 60


def load_gt_test_set():
    """Load Phase 08 ground truth as (query, relevant_tool_names_set) pairs."""
    from src.annotation.label_studio_prep import load_ground_truth
    return load_ground_truth(GT_PATH)


def gt_precision_at_k(results: list[dict], relevant_tools: set, k: int) -> float:
    """Precision@K using exact tool name match against ground truth."""
    top_k = results[:k]
    hits = sum(1 for r in top_k if r["tool_name"] in relevant_tools)
    return hits / len(top_k) if top_k else 0.0


def evaluate_gt(retrieve_fn, test_set, qdrant_path, **retrieve_kwargs) -> float:
    """Evaluate mean Precision@K against real ground truth labels."""
    p5_scores, p10_scores = [], []

    for query, relevant_tools in test_set:
        try:
            results = retrieve_fn(query=query, qdrant_path=qdrant_path, return_k=10, **retrieve_kwargs)
        except Exception:
            results = []

        p5_scores.append(gt_precision_at_k(results, relevant_tools, 5))
        p10_scores.append(gt_precision_at_k(results, relevant_tools, 10))

    p5  = sum(p5_scores)  / len(p5_scores)
    p10 = sum(p10_scores) / len(p10_scores)
    return (p5 + p10) / 2


_cached_test_set = None   # loaded once, reused across all 60 trials


def objective(trial: optuna.Trial) -> float:
    global _cached_test_set
    from src.models.ranker import retrieve

    if _cached_test_set is None:
        _cached_test_set = load_gt_test_set()

    # Three active signals: semantic, BM25, category. Popularity fixed at 0.
    w_semantic = trial.suggest_float("w_semantic", 0.40, 0.90)
    w_bm25     = trial.suggest_float("w_bm25",     0.05, 0.45)
    w_category = trial.suggest_float("w_category", 0.00, 0.20)

    # Normalise so they sum to 1
    total = w_semantic + w_bm25 + w_category
    w_semantic /= total
    w_bm25     /= total
    w_category /= total

    trial.set_user_attr("w_semantic",   round(w_semantic, 4))
    trial.set_user_attr("w_bm25",       round(w_bm25,     4))
    trial.set_user_attr("w_category",   round(w_category, 4))
    trial.set_user_attr("w_popularity", 0.0)

    return evaluate_gt(
        retrieve,
        _cached_test_set,
        qdrant_path=QDRANT_PATH,
        top_k=20,
        w_semantic=w_semantic,
        w_bm25=w_bm25,
        w_category=w_category,
        w_popularity=0.0,
    )


def main():
    from src.models.ranker import retrieve

    if not GT_PATH.exists():
        print("ERROR: data/ground_truth.json not found. Run Phase 08 first.")
        sys.exit(1)

    test_set = load_gt_test_set()

    print("=" * 60)
    print("PHASE 05b — Optuna Re-tuning (Real Ground Truth)")
    print(f"  Ground truth queries: {len(test_set)}")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Metric: mean(P@5, P@10) vs Claude-annotated labels")
    print("=" * 60)

    # ── Baseline (current best_weights from Phase 05) ─────────────────────────
    weights_path = root / "config" / "best_weights.yaml"
    with open(weights_path) as f:
        w = yaml.safe_load(f)

    baseline = evaluate_gt(
        retrieve, test_set,
        qdrant_path=QDRANT_PATH,
        top_k=20,
        w_semantic=w.get("w_semantic", 0.80),
        w_bm25=w.get("w_bm25", 0.15),
        w_category=w.get("w_category", 0.05),
        w_popularity=w.get("w_popularity", 0.00),
    )
    print(f"\nBaseline weights: semantic={w.get('w_semantic',0.80)} | "
          f"bm25={w.get('w_bm25',0.15)} | "
          f"category={w.get('w_category',0.05)} | "
          f"popularity={w.get('w_popularity',0.00)}")
    print(f"Baseline mean P@K (real GT): {baseline:.4f}")

    # ── Optuna ────────────────────────────────────────────────────────────────
    print(f"\nRunning {N_TRIALS} Optuna trials against real ground truth...")
    study = optuna.create_study(
        study_name="toolsense_gt_weight_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best  = study.best_trial
    w_sem = best.user_attrs["w_semantic"]
    w_b25 = best.user_attrs["w_bm25"]
    w_cat = best.user_attrs["w_category"]
    w_pop = 0.0

    print(f"\nBest weights found:")
    print(f"  w_semantic   = {w_sem:.4f}")
    print(f"  w_bm25       = {w_b25:.4f}")
    print(f"  w_category   = {w_cat:.4f}")
    print(f"  w_popularity = {w_pop:.4f}")
    print(f"  sum          = {w_sem + w_b25 + w_cat + w_pop:.4f}")
    print(f"  mean P@K     = {best.value:.4f}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    best_p5s, best_p10s = [], []
    for query, relevant_tools in test_set:
        try:
            results = retrieve(
                query=query, qdrant_path=QDRANT_PATH, top_k=20, return_k=10,
                w_semantic=w_sem, w_bm25=w_b25, w_category=w_cat, w_popularity=w_pop,
            )
        except Exception:
            results = []
        best_p5s.append(gt_precision_at_k(results, relevant_tools, 5))
        best_p10s.append(gt_precision_at_k(results, relevant_tools, 10))

    best_p5  = sum(best_p5s)  / len(best_p5s)
    best_p10 = sum(best_p10s) / len(best_p10s)

    improvement = (best.value - baseline) * 100
    print(f"\nBest weights -> P@5={best_p5:.4f}  P@10={best_p10:.4f}")
    print(f"Improvement over baseline: {improvement:+.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    best_weights = {
        "w_semantic":            round(w_sem, 4),
        "w_bm25":                round(w_b25, 4),
        "w_category":            round(w_cat, 4),
        "w_popularity":          round(w_pop, 4),
        "mean_precision_at_k":   round(best.value, 4),
        "tuned_on":              f"bm25_ground_truth_{len(test_set)}_queries",
    }
    with open(weights_path, "w") as f:
        yaml.dump(best_weights, f, default_flow_style=False)
    print(f"\nBest weights saved to config/best_weights.yaml")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_key = os.getenv("WANDB_API_KEY", "").strip()
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key)
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "toolsense"),
                name="phase05b-optuna-gt-tuning",
                tags=["optuna", "ground-truth", "phase05b"],
            )
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    wandb.log({
                        "trial":        t.number,
                        "w_semantic":   t.params.get("w_semantic", 0),
                        "w_category":   t.params.get("w_category", 0),
                        "w_popularity": t.user_attrs.get("w_popularity", 0),
                        "mean_pk_gt":   t.value,
                    })
            wandb.log({
                "baseline_mean_pk_gt": baseline,
                "best_mean_pk_gt":     best.value,
                "best_p5_gt":          best_p5,
                "best_p10_gt":         best_p10,
                "best_w_semantic":     w_sem,
                "best_w_category":     w_cat,
                "best_w_popularity":   w_pop,
                "improvement_pct":     improvement,
            })
            wandb.finish()
            print("Results logged to W&B")
        except Exception as e:
            print(f"W&B logging failed: {e}")

    print("\n" + "=" * 60)
    print("PHASE 05b COMPLETE")
    print("=" * 60)
    print(f"  Phase 05  weights (proxy GT):  sem={w['w_semantic']} cat={w['w_category']} pop={w['w_popularity']}")
    print(f"  Phase 05b weights (Claude GT): sem={round(w_sem,4)} cat={round(w_cat,4)} pop={round(w_pop,4)}")
    print(f"  P@K improvement: {improvement:+.1f}%")
    print("=" * 60)
    print("Next: re-run Phase 08 to see updated evaluation scores")
    print("  python run_phase08.py --convert")


if __name__ == "__main__":
    main()
