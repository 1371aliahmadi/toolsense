"""
Semantic Cache — Phase 11
==========================
Caches search results in Redis using semantic similarity.

Instead of exact key matching, a new query hits the cache if it is
cosine-similar enough (>= threshold) to a previously cached query.
This avoids re-embedding and re-ranking for near-duplicate queries.

Falls back silently to no-cache mode if Redis is unavailable.

Usage:
    from src.cache.semantic_cache import SemanticCache
    cache = SemanticCache()
    hit = cache.get("AI video generator")
    if hit is None:
        results = retrieve(...)
        cache.set("AI video generator", results)
"""

import json
import logging
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS  = 60 * 60 * 24   # 24 hours
SIMILARITY_THRESHOLD = 0.88          # cosine similarity to consider a cache hit
CACHE_KEY_PREFIX   = "toolsense:query:"
EMBEDDING_FIELD    = "embedding"
RESULTS_FIELD      = "results"
QUERY_FIELD        = "query"
TIMESTAMP_FIELD    = "ts"


class SemanticCache:
    """
    Redis-backed semantic cache for search results.

    Cache entries are stored as Redis hashes:
      key:  toolsense:query:<uuid>
      fields: query (str), embedding (JSON list), results (JSON list), ts (float)

    On `get(query)`:
      1. Embed the query
      2. Scan all cached keys
      3. Compute cosine similarity between query embedding and each cached embedding
      4. If max similarity >= threshold, return the cached results

    On `set(query, results)`:
      Store a new entry with TTL.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        threshold: float = SIMILARITY_THRESHOLD,
        ttl: int = CACHE_TTL_SECONDS,
    ):
        self.threshold = threshold
        self.ttl = ttl
        self._client = None
        self._model = None

        try:
            import redis
            client = redis.Redis(host=host, port=port, db=db,
                                 socket_connect_timeout=2, decode_responses=True)
            client.ping()
            self._client = client
            log.info(f"SemanticCache connected to Redis at {host}:{port}")
        except Exception as e:
            log.warning(f"Redis unavailable ({e}) — running without cache")

    @property
    def available(self) -> bool:
        return self._client is not None

    def _get_model(self):
        """Lazy-load the embedding model (same one used by the ranker)."""
        if self._model is None:
            from src.models.ranker import get_model, load_config
            cfg = load_config()
            self._model = get_model(cfg)
        return self._model

    def _embed(self, query: str) -> np.ndarray:
        model = self._get_model()
        return model.encode(query, normalize_embeddings=True, convert_to_numpy=True)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised vectors."""
        return float(np.dot(a, b))

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[list[dict]]:
        """
        Return cached results for a semantically similar query, or None.
        """
        if not self.available:
            return None

        try:
            query_vec = self._embed(query)
            keys = self._client.keys(f"{CACHE_KEY_PREFIX}*")

            best_sim  = -1.0
            best_key  = None

            for key in keys:
                raw = self._client.hget(key, EMBEDDING_FIELD)
                if raw is None:
                    continue
                cached_vec = np.array(json.loads(raw), dtype=np.float32)
                sim = self._cosine(query_vec, cached_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_sim >= self.threshold and best_key:
                cached_query   = self._client.hget(best_key, QUERY_FIELD) or ""
                results_raw    = self._client.hget(best_key, RESULTS_FIELD)
                results        = json.loads(results_raw) if results_raw else []
                log.info(
                    f"Cache HIT  sim={best_sim:.3f}  "
                    f"'{query[:40]}' ~ '{cached_query[:40]}'"
                )
                return results

            log.debug(f"Cache MISS  best_sim={best_sim:.3f}  '{query[:40]}'")
            return None

        except Exception as e:
            log.warning(f"Cache get error: {e}")
            return None

    def set(self, query: str, results: list[dict]) -> None:
        """
        Store results for a query in the cache with TTL.
        """
        if not self.available:
            return

        try:
            import uuid
            query_vec = self._embed(query)
            key = f"{CACHE_KEY_PREFIX}{uuid.uuid4().hex}"
            self._client.hset(key, mapping={
                QUERY_FIELD:     query,
                EMBEDDING_FIELD: json.dumps(query_vec.tolist()),
                RESULTS_FIELD:   json.dumps(results),
                TIMESTAMP_FIELD: str(time.time()),
            })
            self._client.expire(key, self.ttl)
            log.debug(f"Cache SET  '{query[:40]}'  ({len(results)} results, TTL={self.ttl}s)")

        except Exception as e:
            log.warning(f"Cache set error: {e}")

    def flush(self) -> int:
        """Delete all ToolSense cache entries. Returns number of keys deleted."""
        if not self.available:
            return 0
        try:
            keys = self._client.keys(f"{CACHE_KEY_PREFIX}*")
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            log.warning(f"Cache flush error: {e}")
            return 0

    def stats(self) -> dict:
        """Return cache statistics."""
        if not self.available:
            return {"available": False}
        try:
            keys  = self._client.keys(f"{CACHE_KEY_PREFIX}*")
            info  = self._client.info("memory")
            return {
                "available":      True,
                "n_entries":      len(keys),
                "used_memory_mb": round(info.get("used_memory", 0) / 1e6, 2),
                "threshold":      self.threshold,
                "ttl_hours":      self.ttl // 3600,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
