"""
ToolSense FastAPI Backend -- Phase 09/11
=========================================
REST API wrapping the hybrid ranker.

Endpoints:
  GET  /health          -- liveness check
  GET  /weights         -- current scoring weights
  POST /search          -- semantic tool search (with Redis cache)
  POST /feedback        -- thumbs-up/down from Streamlit UI
  GET  /cache/stats     -- Redis semantic cache stats
  DELETE /cache         -- flush the cache
  GET  /monitor/summary -- quick query log stats
  POST /monitor/report  -- generate Evidently drift report

Start with:
  uvicorn src.api.main:app --reload --port 8000
"""

import time
import logging
import yaml
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parents[2]
QDRANT_PATH  = str(ROOT / "data" / "qdrant")
WEIGHTS_PATH = ROOT / "config" / "best_weights.yaml"

_cache   = None
_monitor = None


def get_cache():
    global _cache
    if _cache is None:
        from src.cache.semantic_cache import SemanticCache
        _cache = SemanticCache(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
        )
    return _cache


def get_monitor():
    global _monitor
    if _monitor is None:
        from src.monitoring.drift_monitor import DriftMonitor
        _monitor = DriftMonitor()
    return _monitor


def _load_weights() -> dict:
    with open(WEIGHTS_PATH) as f:
        w = yaml.safe_load(f)
    return {
        "w_semantic":   float(w.get("w_semantic",   0.40)),
        "w_bm25":       float(w.get("w_bm25",       0.45)),
        "w_category":   float(w.get("w_category",   0.14)),
        "w_popularity": float(w.get("w_popularity", 0.00)),
    }


_WEIGHTS = _load_weights()


class SearchRequest(BaseModel):
    query:           str             = Field(..., min_length=1, max_length=500)
    top_k:           int             = Field(10, ge=1, le=50)
    pricing_tier:    Optional[str]   = Field(None)
    w_semantic:      Optional[float] = Field(None, ge=0, le=1)
    w_bm25:          Optional[float] = Field(None, ge=0, le=1)
    w_category:      Optional[float] = Field(None, ge=0, le=1)
    w_popularity:    Optional[float] = Field(None, ge=0, le=1)
    category_boosts: Optional[dict]  = Field(None)


class ToolResult(BaseModel):
    tool_name:        str
    category:         str
    description:      str
    pricing_tier:     str
    upvotes:          int
    url:              str
    source:           str
    final_score:      float
    semantic_score:   float
    bm25_score:       float
    category_score:   float
    popularity_score: float


class SearchResponse(BaseModel):
    query:           str
    corrected_query: Optional[str]
    results:         list[ToolResult]
    took_ms:         float
    n_results:       int


class FeedbackRequest(BaseModel):
    tool_name: str
    vote:      str
    category:  Optional[str] = None


app = FastAPI(
    title="ToolSense API",
    description="Semantic AI Tool Discovery Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/weights", tags=["Meta"])
def get_weights():
    return _WEIGHTS


@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search(req: SearchRequest):
    from src.models.ranker import retrieve, correct_query

    t0      = time.perf_counter()
    cache   = get_cache()
    monitor = get_monitor()

    corrected       = correct_query(req.query)
    corrected_query = corrected if corrected != req.query else None

    cache_hit = False
    results   = cache.get(req.query)
    if results is not None:
        cache_hit = True
        if req.pricing_tier:
            results = [r for r in results
                       if r.get("pricing_tier", "").lower() == req.pricing_tier.lower()]
        results = results[:req.top_k]

    if not cache_hit:
        weights = {
            "w_semantic":   req.w_semantic   if req.w_semantic   is not None else _WEIGHTS["w_semantic"],
            "w_bm25":       req.w_bm25       if req.w_bm25       is not None else _WEIGHTS["w_bm25"],
            "w_category":   req.w_category   if req.w_category   is not None else _WEIGHTS["w_category"],
            "w_popularity": req.w_popularity if req.w_popularity is not None else _WEIGHTS["w_popularity"],
        }
        filters = {}
        if req.pricing_tier:
            filters["pricing_tier"] = req.pricing_tier
        try:
            results = retrieve(
                query=req.query,
                qdrant_path=QDRANT_PATH,
                top_k=50,
                return_k=req.top_k,
                filters=filters or None,
                category_boosts=req.category_boosts or {},
                **weights,
            )
        except Exception as e:
            log.error("Retrieval error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
        cache.set(req.query, results)

    took_ms = round((time.perf_counter() - t0) * 1000, 1)
    monitor.log(
        query=req.query,
        results=results,
        latency_ms=took_ms,
        corrected=corrected_query is not None,
    )

    return SearchResponse(
        query=req.query,
        corrected_query=corrected_query,
        results=[ToolResult(**r) for r in results],
        took_ms=took_ms,
        n_results=len(results),
    )


@app.post("/feedback", tags=["Search"])
def record_feedback(req: FeedbackRequest):
    """Accept thumbs-up/down from Streamlit UI. Category boost is applied client-side."""
    log.info("Feedback: %s on '%s' (category=%s)", req.vote, req.tool_name, req.category)
    return {"status": "ok", "tool_name": req.tool_name, "vote": req.vote}


@app.get("/cache/stats", tags=["Cache"])
def cache_stats():
    return get_cache().stats()


@app.delete("/cache", tags=["Cache"])
def flush_cache():
    n = get_cache().flush()
    return {"deleted": n}


@app.get("/monitor/summary", tags=["Monitoring"])
def monitor_summary():
    return get_monitor().summary()


@app.post("/monitor/report", tags=["Monitoring"])
def generate_drift_report():
    path = get_monitor().generate_report()
    if path is None:
        raise HTTPException(status_code=500, detail="Report generation failed")
    return {"report_path": str(path)}
