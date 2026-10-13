"""
Phase 04 Runner — Hybrid Recommender + RAG
============================================
Tests the full pipeline end-to-end with sample queries:
  1. Single-turn retrieval (ranker only, no RAG) — sanity check
  2. Single-turn retrieval + RAG explanations
  3. Multi-turn conversation (query → refine → refine)

Logs results to Weights & Biases.

Usage (from toolsense/ directory):
    python run_phase04.py

Requirements:
  - Phase 03 must have run (data/qdrant/ must exist)
  - Ollama must be running: `ollama serve` + `ollama pull llama3.2:3b`
"""

import sys
import json
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

# ── Test queries ──────────────────────────────────────────────────────────────

SINGLE_QUERIES = [
    "I want to generate AI-powered videos for my YouTube channel",
    "help me write better emails and marketing copy",
    "I need a chatbot for my customer support team",
    "tool for generating images from text descriptions",
    "automate my coding workflow and fix bugs automatically",
]

CONVERSATION_FLOW = [
    "I need a tool to create AI videos",
    "show me only free ones",
    "which is the most popular?",
]


def print_results(results: list[dict], show_explanation: bool = True) -> None:
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['tool_name']}  [{r['final_score']:.3f}]")
        print(f"     Category: {r['category']} | Pricing: {r['pricing_tier']} | Upvotes: {r['upvotes']}")
        print(f"     Scores -> semantic: {r['semantic_score']} | category: {r['category_score']} | popularity: {r['popularity_score']}")
        if show_explanation and "explanation" in r:
            print(f"     [AI] {r['explanation']}")
        print(f"     [URL] {r['url']}")


def main():
    qdrant_path = str(root / "data" / "qdrant")

    if not Path(qdrant_path).exists():
        log.error("data/qdrant/ not found. Run Phase 03 first.")
        sys.exit(1)

    from src.models.ranker import retrieve
    from src.models.rag import add_explanations
    from src.models.conversation import ToolSenseConversation

    # ── Test 1: Ranker only (no RAG) ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 1 — Ranker only (no LLM explanations)")
    print("=" * 60)

    ranker_results_log = []
    for query in SINGLE_QUERIES:
        print(f"\nQuery: '{query}'")
        results = retrieve(query=query, qdrant_path=qdrant_path)
        print_results(results, show_explanation=False)
        ranker_results_log.append({
            "query": query,
            "top_tool": results[0]["tool_name"] if results else "none",
            "top_score": results[0]["final_score"] if results else 0,
        })

    # ── Test 2: Full RAG pipeline ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2 — Full RAG pipeline (with LLM explanations)")
    print("=" * 60)

    query = SINGLE_QUERIES[0]
    print(f"\nQuery: '{query}'")
    results = retrieve(query=query, qdrant_path=qdrant_path, return_k=3)
    results = add_explanations(results, query)
    print_results(results, show_explanation=True)

    # ── Test 3: Multi-turn conversation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3 — Multi-turn conversation")
    print("=" * 60)

    session = ToolSenseConversation(qdrant_path=qdrant_path, use_rag=False)
    for turn in CONVERSATION_FLOW:
        print(f"\nUser: {turn}")
        results = session.ask(turn)
        print(f"Found {len(results)} tools:")
        for r in results:
            print(f"   • {r['tool_name']} ({r['pricing_tier']}) — score: {r['final_score']:.3f}")

    # ── Log to W&B ────────────────────────────────────────────────────────────
    wandb_key = os.getenv("WANDB_API_KEY", "").strip()
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key)
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "toolsense"),
                name="phase04-recommender-rag",
                tags=["recommender", "rag", "phase04"],
            )

            # Log retrieval results as a W&B Table
            table = wandb.Table(columns=["query", "top_tool", "top_score"])
            for row in ranker_results_log:
                table.add_data(row["query"], row["top_tool"], row["top_score"])
            wandb.log({"retrieval_results": table})

            wandb.log({
                "queries_tested": len(SINGLE_QUERIES),
                "conversation_turns": len(CONVERSATION_FLOW),
                "w_semantic": 0.60,
                "w_category": 0.25,
                "w_popularity": 0.15,
            })

            wandb.finish()
            log.info("Results logged to W&B ✓")
        except Exception as e:
            log.warning(f"W&B logging failed: {e}")

    print("\n" + "=" * 60)
    print("PHASE 04 COMPLETE")
    print("  Ranker: OK  |  RAG: OK  |  Conversation: OK")
    print("=" * 60)
    print("Next step: Phase 05 — Optuna Hyperparameter Tuning")


if __name__ == "__main__":
    main()
