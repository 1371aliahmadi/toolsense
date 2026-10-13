"""
Phase 03 Runner — Vector Embeddings
=====================================
Embeds clean_tools.csv with two sentence-transformer models,
stores the active model's vectors in a local Qdrant database,
and logs a model comparison to Weights & Biases.

Usage (from toolsense/ directory):
    python run_phase03.py
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
    clean_path  = root / "data" / "clean_tools.csv"
    qdrant_path = root / "data" / "qdrant"

    if not clean_path.exists():
        log.error("data/clean_tools.csv not found. Run Phase 02 first.")
        sys.exit(1)

    # ── Run embedding pipeline ────────────────────────────────────────────────
    from src.embeddings.generator import run_embedding_pipeline
    metrics = run_embedding_pipeline(
        clean_path=clean_path,
        qdrant_path=qdrant_path,
    )

    # ── Log to W&B ────────────────────────────────────────────────────────────
    wandb_key = os.getenv("WANDB_API_KEY", "").strip()
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key)
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "toolsense"),
                name="phase03-embeddings",
                tags=["embeddings", "phase03"],
            )

            wandb.log({
                "tool_count":         metrics["tool_count"],
                "active_model":       metrics["active_model_name"],
                "model_a_dim":        metrics["model_a_dim"],
                "model_a_time_s":     metrics["model_a_time_s"],
                "model_b_dim":        metrics["model_b_dim"],
                "model_b_time_s":     metrics["model_b_time_s"],
                "vectors_stored":     metrics["vectors_stored"],
                "vector_dim":         metrics["vector_dim"],
            })

            # W&B Table for model comparison
            table = wandb.Table(columns=["model", "dimensions", "time_seconds"])
            table.add_data(metrics["model_a_name"], metrics["model_a_dim"], metrics["model_a_time_s"])
            table.add_data(metrics["model_b_name"], metrics["model_b_dim"], metrics["model_b_time_s"])
            wandb.log({"model_comparison": table})

            wandb.finish()
            log.info("Metrics logged to W&B ✓")
        except Exception as e:
            log.warning(f"W&B logging failed: {e} — continuing without it")
    else:
        log.warning("WANDB_API_KEY not set — skipping W&B logging.")

    log.info("\n" + "=" * 60)
    log.info("PHASE 03 COMPLETE")
    log.info(f"  Tools embedded:   {metrics['tool_count']}")
    log.info(f"  Active model:     {metrics['active_model_name']} ({metrics['vector_dim']}-dim)")
    log.info(f"  Vectors in Qdrant: {metrics['vectors_stored']}")
    log.info(f"  Qdrant path:      data/qdrant/")
    log.info("=" * 60)
    log.info("Next step: Phase 04 — Hybrid Recommender + RAG")


if __name__ == "__main__":
    main()
