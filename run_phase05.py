"""
Phase 05 Runner — Optuna Hyperparameter Tuning
================================================
Tunes the hybrid scoring weights (w_semantic, w_category, w_popularity)
using Optuna to maximise Precision@K on a proxy test set.

Constraint: weights must sum to 1.0

Usage (from toolsense/ directory):
    python run_phase05.py

Outputs:
  - Best weights printed + saved to config/best_weights.yaml
  - All trial results logged to W&B
  - Optuna study visualisation saved to data/optuna_study.html
"""

import sys
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
warnings.filterwarnings("ignore")  # suppress Optuna/sklearn noise

logging.basicConfig(
    level=logging.WARNING,   # quiet during tuning — only show important messages
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)
logging.getLogger("src").setLevel(logging.WARNING)

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

import yaml
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


QDRANT_PATH = str(root / "data" / "qdrant")
N_TRIALS    = 60   # fast enough (~3-5 min), enough to find good weights


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    Samples w1, w2, w3 such that w1 + w2 + w3 = 1.0
    Returns mean Precision@K (higher = better).
    """
    from src.models.ranker import retrieve
    from src.evaluation.metrics import evaluate

    # Sample two values; third is the remainder
    w_semantic  = trial.suggest_float("w_semantic",  0.30, 0.80)
    w_category  = trial.suggest_float("w_category",  0.05, 0.40)
    w_popularity = max(0.0, 1.0 - w_semantic - w_category)

    # Prune if weights are invalid
    if w_popularity < 0 or w_popularity > 0.50:
        raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("w_popularity", round(w_popularity, 4))

    result = evaluate(
        retrieve_fn=retrieve,
        qdrant_path=QDRANT_PATH,
        top_k=20,
        return_k=10,
        w_semantic=w_semantic,
        w_category=w_category,
        w_popularity=w_popularity,
    )

    return result.mean_score


def main():
    from src.models.ranker import retrieve
    from src.evaluation.metrics import evaluate, PROXY_TEST_SET

    print("=" * 60)
    print("PHASE 05 — Optuna Hyperparameter Tuning")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Test set: {len(PROXY_TEST_SET)} queries")
    print(f"  Metric: mean(Precision@5, Precision@10)")
    print("=" * 60)

    # ── Baseline evaluation (starting weights from config) ────────────────────
    cfg_path = root / "config" / "base_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    baseline = evaluate(
        retrieve_fn=retrieve,
        qdrant_path=QDRANT_PATH,
        top_k=20,
        return_k=10,
        w_semantic=cfg["ranker"]["w_semantic"],
        w_category=cfg["ranker"]["w_category"],
        w_popularity=cfg["ranker"]["w_popularity"],
    )
    print(f"\nBaseline weights: semantic={cfg['ranker']['w_semantic']} | "
          f"category={cfg['ranker']['w_category']} | "
          f"popularity={cfg['ranker']['w_popularity']}")
    print(f"Baseline  ->  P@5={baseline.precision_at_5:.4f}  "
          f"P@10={baseline.precision_at_10:.4f}  "
          f"mean={baseline.mean_score:.4f}")

    # ── Optuna study ──────────────────────────────────────────────────────────
    print(f"\nRunning {N_TRIALS} Optuna trials...")
    study = optuna.create_study(
        study_name="toolsense_weight_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    w_sem = best.params["w_semantic"]
    w_cat = best.params["w_category"]
    w_pop = best.user_attrs["w_popularity"]

    print(f"\nBest weights found:")
    print(f"  w_semantic   = {w_sem:.4f}")
    print(f"  w_category   = {w_cat:.4f}")
    print(f"  w_popularity = {w_pop:.4f}")
    print(f"  sum          = {w_sem + w_cat + w_pop:.4f}")
    print(f"  mean P@K     = {best.value:.4f}")

    # ── Evaluate best weights ─────────────────────────────────────────────────
    best_eval = evaluate(
        retrieve_fn=retrieve,
        qdrant_path=QDRANT_PATH,
        top_k=20,
        return_k=10,
        w_semantic=w_sem,
        w_category=w_cat,
        w_popularity=w_pop,
    )
    print(f"\nBest weights ->  P@5={best_eval.precision_at_5:.4f}  "
          f"P@10={best_eval.precision_at_10:.4f}  "
          f"mean={best_eval.mean_score:.4f}")
    print(f"Improvement over baseline: "
          f"{(best_eval.mean_score - baseline.mean_score) * 100:+.1f}%")

    # ── Save best weights ─────────────────────────────────────────────────────
    best_weights = {
        "w_semantic":   round(w_sem, 4),
        "w_category":   round(w_cat, 4),
        "w_popularity": round(w_pop, 4),
        "mean_precision_at_k": round(best_eval.mean_score, 4),
    }
    weights_path = root / "config" / "best_weights.yaml"
    with open(weights_path, "w") as f:
        yaml.dump(best_weights, f, default_flow_style=False)
    print(f"\nBest weights saved to config/best_weights.yaml")

    # ── Save Optuna visualisation ─────────────────────────────────────────────
    try:
        from optuna.visualization import plot_param_importances, plot_optimization_history
        import plotly.io as pio

        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

        out_dir = root / "data"
        fig1.write_html(str(out_dir / "optuna_history.html"))
        fig2.write_html(str(out_dir / "optuna_importances.html"))
        print("Optuna plots saved to data/optuna_history.html and data/optuna_importances.html")
    except Exception as e:
        print(f"(Optuna plots skipped: {e})")

    # ── Log to W&B ────────────────────────────────────────────────────────────
    wandb_key = os.getenv("WANDB_API_KEY", "").strip()
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key)
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "toolsense"),
                name="phase05-optuna-tuning",
                tags=["optuna", "tuning", "phase05"],
            )

            # Log all trials
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    w_p = t.user_attrs.get("w_popularity", 0)
                    wandb.log({
                        "trial":        t.number,
                        "w_semantic":   t.params.get("w_semantic", 0),
                        "w_category":   t.params.get("w_category", 0),
                        "w_popularity": w_p,
                        "mean_pk":      t.value,
                    })

            wandb.log({
                "baseline_mean_pk":      baseline.mean_score,
                "best_mean_pk":          best_eval.mean_score,
                "best_w_semantic":       w_sem,
                "best_w_category":       w_cat,
                "best_w_popularity":     w_pop,
                "best_precision_at_5":   best_eval.precision_at_5,
                "best_precision_at_10":  best_eval.precision_at_10,
            })

            wandb.finish()
            print("Results logged to W&B")
        except Exception as e:
            print(f"W&B logging failed: {e}")

    print("\n" + "=" * 60)
    print("PHASE 05 COMPLETE")
    print("=" * 60)
    print("Next step: Phase 06 — Performance Evaluation")


if __name__ == "__main__":
    main()
