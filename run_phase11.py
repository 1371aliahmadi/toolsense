"""
Phase 11 Runner — Redis Semantic Cache + Evidently Drift Monitoring
====================================================================
Validates Redis connection, warms up the cache, logs synthetic
requests, and generates an Evidently drift report.

Usage:
    python run_phase11.py
"""

import sys
import json
import time
import logging
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

QDRANT_PATH = str(root / "data" / "qdrant")
REPORT_PATH = root / "data" / "drift_report.html"

# Sample queries to warm cache + generate log entries
SAMPLE_QUERIES = [
    "AI writing assistant for blog posts",
    "generate videos from text automatically",
    "customer support chatbot builder",
    "SEO keyword research and optimization",
    "code review and debugging AI",
    "transcribe meetings to text",
    "AI image generation from prompts",
    "automate email marketing campaigns",
    "HR recruiting and resume screening",
    "translate documents between languages",
    "create slide decks from text",
    "data analysis and business intelligence",
    "AI tools for e-commerce product descriptions",
    "social media content scheduler",
    "background removal from photos",
    "AI voice cloning and text to speech",
    "contract review and legal document analysis",
    "generate unit tests for Python code",
    "AI for financial forecasting",
    "no-code app builder with AI",
    # Slightly rephrased versions (should hit cache)
    "writing assistant for blogs",
    "video generation tool",
    "chatbot for customer service",
    "SEO content planning",
    "debug Python code with AI",
]


def main():
    print("=" * 60)
    print("PHASE 11 -- Redis Cache + Evidently Monitoring")
    print("=" * 60)

    from src.cache.semantic_cache import SemanticCache
    from src.monitoring.drift_monitor import DriftMonitor
    from src.models.ranker import retrieve
    import yaml

    weights_path = root / "config" / "best_weights.yaml"
    with open(weights_path) as f:
        w = yaml.safe_load(f)
    weights = {
        "w_semantic":   w["w_semantic"],
        "w_category":   w["w_category"],
        "w_popularity": w["w_popularity"],
    }

    # ── 1. Cache setup ─────────────────────────────────────────────────────────
    print("\n[1] Redis Semantic Cache")
    cache = SemanticCache(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )
    if cache.available:
        print("  Redis connected!")
        cache.flush()
        print("  Cache flushed (fresh start)")
    else:
        print("  Redis not available — running without cache (still logs for monitoring)")
        print("  To enable: docker run -d -p 6379:6379 redis:alpine")

    # ── 2. Run sample queries: fill cache + log ────────────────────────────────
    print(f"\n[2] Running {len(SAMPLE_QUERIES)} sample queries...")

    monitor = DriftMonitor()
    cache_hits = 0

    for i, query in enumerate(SAMPLE_QUERIES):
        t0 = time.perf_counter()

        # Check cache
        results = cache.get(query)
        hit = results is not None

        if not hit:
            results = retrieve(
                query=query,
                qdrant_path=QDRANT_PATH,
                return_k=10,
                **weights,
            )
            cache.set(query, results)
        else:
            cache_hits += 1

        latency = (time.perf_counter() - t0) * 1000
        monitor.log(query=query, results=results, latency_ms=latency)

        tag = "HIT " if hit else "MISS"
        score = results[0]["final_score"] if results else 0.0
        print(f"  [{i+1:02d}/{len(SAMPLE_QUERIES)}] [{tag}] {query[:45]:<45}  "
              f"top={score:.3f}  {latency:.0f}ms")

    # ── 3. Cache stats ─────────────────────────────────────────────────────────
    print("\n[3] Cache Stats")
    stats = cache.stats()
    if stats.get("available"):
        hit_rate = round(cache_hits / len(SAMPLE_QUERIES) * 100, 1)
        print(f"  Entries:      {stats['n_entries']}")
        print(f"  Memory:       {stats['used_memory_mb']} MB")
        print(f"  Cache hits:   {cache_hits}/{len(SAMPLE_QUERIES)} ({hit_rate}%)")
        print(f"  Threshold:    {stats['threshold']}")
        print(f"  TTL:          {stats['ttl_hours']}h")
    else:
        print("  Cache unavailable (Redis not running)")

    # ── 4. Monitor summary ─────────────────────────────────────────────────────
    print("\n[4] Query Log Summary")
    summary = monitor.summary()
    print(f"  Total logged:      {summary.get('n_entries', 0)}")
    print(f"  Mean top score:    {summary.get('mean_top_score', 0)}")
    print(f"  Mean latency:      {summary.get('mean_latency_ms', 0)} ms")
    print(f"  P95 latency:       {summary.get('p95_latency_ms', 0)} ms")
    print(f"  Top categories:    {', '.join(summary.get('top_categories', []))}")

    # ── 5. Evidently drift report ──────────────────────────────────────────────
    print("\n[5] Generating Evidently Drift Report...")
    report_path = monitor.generate_report(REPORT_PATH)
    if report_path:
        print(f"  Report saved: {report_path}")
        print("  Open in browser to inspect score/latency/category drift.")
    else:
        n = monitor.n_entries()
        print(f"  Not enough entries yet ({n} logged, need 20).")
        print("  Run more queries via the Streamlit UI, then re-run this script.")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PHASE 11 COMPLETE")
    print("=" * 60)
    print(f"  Cache backend:    {'Redis' if cache.available else 'Disabled (no Redis)'}")
    print(f"  Log entries:      {monitor.n_entries()}")
    print(f"  Drift report:     {REPORT_PATH if report_path else 'not generated yet'}")
    print()
    print("New API endpoints (restart the API server to activate):")
    print("  GET  /cache/stats       -- cache statistics")
    print("  DELETE /cache           -- flush cache")
    print("  GET  /monitor/summary   -- query log stats")
    print("  POST /monitor/report    -- generate drift report")
    print("=" * 60)
    print("Next step: Phase 12 -- Final report + project submission")


if __name__ == "__main__":
    main()
