"""
Phase 02 Runner — Data Quality
================================
Validates raw_tools.csv and produces:
  - data/clean_tools.csv       (validated, deduplicated)
  - data/quality_report.html   (ydata-profiling report)

Logs all quality metrics to Weights & Biases.

Usage (from toolsense/ directory):
    python run_phase02.py
"""

import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))


def main():
    raw_path   = root / "data" / "raw_tools.csv"
    clean_path = root / "data" / "clean_tools.csv"
    report_path = root / "data" / "quality_report.html"

    if not raw_path.exists():
        log.error("data/raw_tools.csv not found. Run Phase 01 first.")
        sys.exit(1)

    # ── Run quality pipeline ──────────────────────────────────────────────────
    from src.data_pipeline.validator import run_quality_pipeline
    df, metrics = run_quality_pipeline(
        raw_path=raw_path,
        clean_path=clean_path,
        report_path=report_path,
    )

    # ── Log to W&B ────────────────────────────────────────────────────────────
    wandb_key = os.getenv("WANDB_API_KEY", "").strip()
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key)
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "toolsense"),
                name="phase02-data-quality",
                tags=["data-quality", "phase02"],
            )
            src = metrics["source_balance"]["counts"]
            after_null = metrics["raw_count"] - metrics["removed_null"]
            wandb.log({
                "raw_tool_count":           metrics["raw_count"],
                "clean_tool_count":         metrics["clean_count"],
                "removed_null":             metrics["removed_null"],
                "removed_template_nsfw":    after_null - metrics.get("after_template_filter", after_null),
                "removed_short_desc":       metrics.get("after_template_filter", after_null) - metrics["after_length_filter"],
                "removed_near_dupes":       metrics["after_length_filter"] - metrics["after_dedup"],
                "ge_passed":                int(metrics["ge_passed"]),
                "source_balance_ok":        int(metrics["source_balance"]["balanced"]),
                "popularity_coverage":      metrics["popularity"]["coverage"],
                "producthunt_count":        src.get("producthunt", 0),
                "huggingface_count":        src.get("huggingface", 0),
                "taaft_count":              src.get("taaft", 0),
                "cross_source_count":       sum(v for k, v in src.items() if "+" in k),
            })

            # Upload quality report as W&B artifact
            artifact = wandb.Artifact("quality_report", type="data-quality")
            artifact.add_file(str(report_path))
            wandb.log_artifact(artifact)

            wandb.finish()
            log.info("Metrics logged to W&B ✓")
        except Exception as e:
            log.warning(f"W&B logging failed: {e} — continuing without it")
    else:
        log.warning(
            "WANDB_API_KEY not set in .env — skipping W&B logging.\n"
            "Sign up at wandb.ai and add your key to .env to enable logging."
        )

    log.info("\n" + "=" * 60)
    log.info("PHASE 02 COMPLETE")
    log.info(f"  Clean dataset:    data/clean_tools.csv ({metrics['clean_count']} tools)")
    log.info(f"  Quality report:   data/quality_report.html")
    log.info("=" * 60)
    log.info("Next step: Phase 03 — Vector Embeddings")


if __name__ == "__main__":
    main()
