"""
Phase 08 Runner -- Label Studio Data Annotation
=================================================
Two modes:

  Mode 1 (default): Generate annotation tasks
    python run_phase08.py
    -> Creates data/label_studio_tasks.json (import this into Label Studio)
    -> Prints Label Studio setup instructions

  Mode 2: Convert export to ground truth
    python run_phase08.py --convert
    -> Reads data/annotations_export.json (exported from Label Studio)
    -> Writes data/ground_truth.json
    -> Re-evaluates with real ground truth and logs to W&B

Label Studio install:
    pip install label-studio
    label-studio start          # opens browser at http://localhost:8080

Usage:
    python run_phase08.py              # step 1: generate tasks
    python run_phase08.py --convert    # step 2: convert after labeling
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

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

QDRANT_PATH   = str(root / "data" / "qdrant")
TASKS_PATH    = root / "data" / "label_studio_tasks.json"
EXPORT_PATH   = root / "data" / "annotations_export.json"
GT_PATH       = root / "data" / "ground_truth.json"
REPORT_PATH   = root / "data" / "eval_report_phase08.html"


def mode_generate():
    """Generate Label Studio import tasks."""
    import yaml
    from src.models.ranker import retrieve
    from src.annotation.label_studio_prep import (
        ANNOTATION_QUERIES, generate_tasks, save_tasks, LABEL_CONFIG
    )

    print("=" * 60)
    print("PHASE 08 -- Label Studio Task Generation")
    print("=" * 60)
    print(f"  Queries to annotate: {len(ANNOTATION_QUERIES)}")
    print(f"  Candidates per query: 20 (top-K retrieved tools)")
    print()

    # Load best weights
    weights_path = root / "config" / "best_weights.yaml"
    with open(weights_path) as f:
        w = yaml.safe_load(f)
    weights = {
        "w_semantic":   w["w_semantic"],
        "w_category":   w["w_category"],
        "w_popularity": w["w_popularity"],
    }

    print("Retrieving top-20 candidates for each query...")
    tasks = generate_tasks(retrieve, QDRANT_PATH, weights, top_k=20)
    save_tasks(tasks, TASKS_PATH)

    print(f"\nTasks saved: {TASKS_PATH}")
    print()
    print("=" * 60)
    print("LABEL STUDIO SETUP INSTRUCTIONS")
    print("=" * 60)
    print()
    print("Step 1 -- Install Label Studio (if not done):")
    print("  pip install label-studio")
    print()
    print("Step 2 -- Start Label Studio:")
    print("  label-studio start")
    print("  -> Opens at http://localhost:8080")
    print()
    print("Step 3 -- Create a new project:")
    print("  Name: ToolSense Annotation")
    print("  Labeling type: Custom template")
    print()
    print("Step 4 -- Paste this labeling config (XML):")
    print("-" * 40)
    print("""<View>
  <Header value="$query"
    style="font-size:1.3em; font-weight:bold; color:#2c3e50; padding:10px 0"/>
  <Text name="instruction" value="$instruction"
    style="color:#7f8c8d; font-style:italic; margin-bottom:8px"/>
  <Text name="ref" value="$candidates_text"
    style="white-space:pre-wrap; font-size:0.82em; color:#555;
           background:#f8f9fa; padding:10px; border-radius:6px;
           max-height:300px; overflow-y:auto; margin-bottom:12px"/>
  <Choices name="relevant_tools" toName="ref"
           choice="multiple" dynValue="true" allowEmpty="true"
           style="max-height:320px; overflow-y:auto">
    <Choice value="$choices"/>
  </Choices>
</View>""")
    print("-" * 40)
    print()
    print("Step 5 -- Import tasks:")
    print(f"  Import > JSON > select: {TASKS_PATH}")
    print()
    print("Step 6 -- Annotate:")
    print("  For each query, tick ALL tools that a user would find relevant.")
    print("  Aim for 50-100 queries labeled.")
    print()
    print("Step 7 -- Export:")
    print("  Export > JSON > save as:")
    print(f"  {EXPORT_PATH}")
    print()
    print("Step 8 -- Convert to ground truth:")
    print("  python run_phase08.py --convert")
    print("=" * 60)


def mode_convert():
    """Convert Label Studio export to ground truth and re-evaluate."""
    from src.models.ranker import retrieve
    from src.annotation.label_studio_prep import (
        convert_export, save_ground_truth, load_ground_truth
    )
    from src.evaluation.evaluator import run_full_evaluation, save_eval_report

    print("=" * 60)
    print("PHASE 08 -- Converting Annotations to Ground Truth")
    print("=" * 60)

    if not EXPORT_PATH.exists():
        print(f"ERROR: Export file not found: {EXPORT_PATH}")
        print("Export your annotations from Label Studio first (Step 7 above).")
        sys.exit(1)

    # Convert
    print(f"Reading export: {EXPORT_PATH}")
    ground_truth = convert_export(EXPORT_PATH)
    save_ground_truth(ground_truth, GT_PATH)

    # Build test set in same format as PROXY_TEST_SET
    test_set = load_ground_truth(GT_PATH)
    print(f"\nGround truth loaded: {len(test_set)} labeled queries")

    if len(test_set) < 10:
        print("WARNING: fewer than 10 labeled queries — results may not be reliable.")
        print("Consider labeling more queries before re-evaluating.")

    # Re-evaluate with real ground truth
    print("\nRe-running evaluation with real ground truth...")
    import yaml
    weights_path = root / "config" / "best_weights.yaml"
    with open(weights_path) as f:
        w = yaml.safe_load(f)

    # Override is_relevant for ground truth mode (exact tool_name match)
    from src.evaluation import metrics as metrics_mod
    original_is_relevant = metrics_mod.is_relevant

    def gt_is_relevant(result: dict, keywords: set[str]) -> bool:
        """For ground truth: keywords IS the set of relevant tool names."""
        return result["tool_name"] in keywords

    metrics_mod.is_relevant = gt_is_relevant

    metrics = run_full_evaluation(
        retrieve_fn=retrieve,
        qdrant_path=QDRANT_PATH,
        clean_csv_path=str(root / "data" / "clean_tools.csv"),
        test_set=test_set,
    )

    metrics_mod.is_relevant = original_is_relevant  # restore

    # Save report
    save_eval_report(metrics, REPORT_PATH)
    print(f"\nEvaluation report: {REPORT_PATH}")

    # Log to W&B
    wandb_key = os.getenv("WANDB_API_KEY", "").strip()
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key)
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "toolsense"),
                name="phase08-ground-truth-eval",
                tags=["annotation", "ground-truth", "phase08"],
                config={
                    "n_labeled_queries": len(test_set),
                    "w_semantic": w["w_semantic"],
                    "w_category": w["w_category"],
                },
            )
            wandb.log({
                "gt_eval/precision_at_5":  metrics["mean_precision_at_5"],
                "gt_eval/precision_at_10": metrics["mean_precision_at_10"],
                "gt_eval/ndcg_at_5":       metrics["mean_ndcg_at_5"],
                "gt_eval/ndcg_at_10":      metrics["mean_ndcg_at_10"],
                "gt_eval/coverage_at_10":  metrics["coverage_at_10"],
                "gt_eval/n_queries":       len(test_set),
            })
            import pandas as pd
            wandb.log({"gt_eval/per_query": wandb.Table(dataframe=metrics["per_query_df"])})
            wandb.finish()
            print("Results logged to W&B (run: phase08-ground-truth-eval)")
        except Exception as e:
            print(f"W&B logging failed: {e}")

    print("\n" + "=" * 60)
    print("PHASE 08 COMPLETE")
    print("=" * 60)
    print(f"  Real ground truth queries: {len(test_set)}")
    print(f"  Precision@5  (real GT):    {metrics['mean_precision_at_5']:.4f}")
    print(f"  Precision@10 (real GT):    {metrics['mean_precision_at_10']:.4f}")
    print(f"  Recall@5     (real GT):    {metrics.get('mean_recall_at_5', 0):.4f}")
    print(f"  Recall@10    (real GT):    {metrics.get('mean_recall_at_10', 0):.4f}")
    print(f"  NDCG@5       (real GT):    {metrics['mean_ndcg_at_5']:.4f}")
    print(f"  Coverage@10:               {metrics.get('coverage_at_10', 0):.4f}")
    print("=" * 60)
    print("Next step: Phase 09 -- FastAPI + Streamlit UI")


def main():
    if "--convert" in sys.argv:
        mode_convert()
    else:
        mode_generate()


if __name__ == "__main__":
    main()
