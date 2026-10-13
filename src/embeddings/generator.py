"""
Embedding Generator — Phase 03
================================
Embeds tool descriptions using sentence-transformers and stores them
in a local Qdrant vector database (no Docker required).

Two models are benchmarked:
  - model_a: all-MiniLM-L6-v2   (384-dim, fast)
  - model_b: all-mpnet-base-v2  (768-dim, higher quality, slower)

The active model (from config) is stored in Qdrant.
Both models' timing + stats are logged to W&B for comparison.

Usage:
    python -m src.embeddings.generator
"""

import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "base_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Model names ───────────────────────────────────────────────────────────────

MODEL_NAMES = {
    "model_a": "all-MiniLM-L6-v2",
    "model_b": "all-mpnet-base-v2",
}


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_descriptions(
    descriptions: list[str],
    model_key: str,
    batch_size: int = 64,
) -> tuple[np.ndarray, float]:
    """
    Embed a list of descriptions with the given model.

    Returns:
        (vectors, elapsed_seconds)
    """
    model_name = MODEL_NAMES[model_key]
    log.info(f"Loading model: {model_name} …")
    model = SentenceTransformer(model_name)

    log.info(f"Embedding {len(descriptions)} descriptions (batch_size={batch_size}) …")
    t0 = time.time()
    vectors = model.encode(
        descriptions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors → cosine = dot product
    )
    elapsed = time.time() - t0
    log.info(f"  Done in {elapsed:.1f}s — shape: {vectors.shape}")
    return vectors, elapsed


# ── Qdrant storage ────────────────────────────────────────────────────────────

def store_in_qdrant(
    df: pd.DataFrame,
    vectors: np.ndarray,
    collection_name: str,
    qdrant_path: str,
    hnsw_m: int = 16,
    hnsw_ef_construct: int = 200,
) -> QdrantClient:
    """
    Create (or recreate) a Qdrant collection and upload all vectors.

    Uses local file-based storage — no Docker required.
    """
    dim = vectors.shape[1]
    log.info(f"Connecting to local Qdrant at: {qdrant_path}")
    client = QdrantClient(path=qdrant_path)

    # Delete existing collection if it exists (handles dim changes between runs)
    if client.collection_exists(collection_name):
        log.info(f"Deleting existing collection '{collection_name}' (dim may have changed) …")
        client.delete_collection(collection_name)

    log.info(f"Creating collection '{collection_name}' (dim={dim}, cosine) …")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
            hnsw_config={
                "m": hnsw_m,
                "ef_construct": hnsw_ef_construct,
            },
        ),
    )

    # Build points
    log.info("Uploading vectors …")
    points = []
    for i, (_, row) in enumerate(df.iterrows()):
        payload = {
            "tool_id":      int(row.get("tool_id", i)),
            "tool_name":    str(row["tool_name"]),
            "description":  str(row["description"]),
            "category":     str(row.get("category", "")),
            "pricing_tier": str(row.get("pricing_tier", "Unknown")),
            "upvotes":      int(row.get("upvotes", 0)),
            "source":       str(row.get("source", "")),
            "url":          str(row.get("url", "")),
        }
        points.append(PointStruct(id=i, vector=vectors[i].tolist(), payload=payload))

    # Upload in batches of 256
    batch_size = 256
    for start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[start : start + batch_size],
        )

    count = client.count(collection_name=collection_name).count
    log.info(f"Qdrant collection '{collection_name}': {count} vectors stored ✓")
    return client


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test(client: QdrantClient, collection_name: str, query_vector: np.ndarray) -> None:
    """Run one test query to confirm the index works."""
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=3,
    )
    log.info("Smoke test — top 3 results for a random query vector:")
    for r in response.points:
        log.info(f"  [{r.score:.3f}] {r.payload['tool_name']} — {r.payload['description'][:60]}…")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_embedding_pipeline(
    clean_path: str | Path = "data/clean_tools.csv",
    qdrant_path: str | Path = "data/qdrant",
) -> dict:
    """
    Full embedding pipeline:
      1. Load clean dataset
      2. Embed with model_a and model_b (for W&B comparison)
      3. Store active model's vectors in Qdrant
      4. Smoke test

    Returns metrics dict for W&B logging.
    """
    cfg = load_config()
    clean_path  = Path(clean_path)
    qdrant_path = str(Path(qdrant_path))

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info(f"Loading {clean_path} …")
    df = pd.read_csv(clean_path)
    log.info(f"  {len(df)} tools loaded")

    descriptions = df["description"].fillna("").tolist()

    batch_size   = cfg["embeddings"]["batch_size"]
    active_model = cfg["embeddings"]["active_model"]   # "model_a" or "model_b"

    metrics = {
        "tool_count": len(df),
        "active_model": active_model,
        "active_model_name": MODEL_NAMES[active_model],
    }

    # ── Embed with both models (for comparison) ───────────────────────────────
    vectors_a, time_a = embed_descriptions(descriptions, "model_a", batch_size)
    vectors_b, time_b = embed_descriptions(descriptions, "model_b", batch_size)

    metrics["model_a_name"]      = MODEL_NAMES["model_a"]
    metrics["model_a_dim"]       = vectors_a.shape[1]
    metrics["model_a_time_s"]    = round(time_a, 2)

    metrics["model_b_name"]      = MODEL_NAMES["model_b"]
    metrics["model_b_dim"]       = vectors_b.shape[1]
    metrics["model_b_time_s"]    = round(time_b, 2)

    log.info(f"\n── Model comparison ──")
    log.info(f"  model_a ({MODEL_NAMES['model_a']}): {vectors_a.shape[1]}-dim, {time_a:.1f}s")
    log.info(f"  model_b ({MODEL_NAMES['model_b']}): {vectors_b.shape[1]}-dim, {time_b:.1f}s")
    log.info(f"  Active model: {active_model} ({MODEL_NAMES[active_model]})")

    # ── Store active model in Qdrant ──────────────────────────────────────────
    active_vectors = vectors_a if active_model == "model_a" else vectors_b

    collection_name = cfg["qdrant"]["collection_name"]
    client = store_in_qdrant(
        df=df,
        vectors=active_vectors,
        collection_name=collection_name,
        qdrant_path=qdrant_path,
        hnsw_m=cfg["qdrant"]["hnsw_m"],
        hnsw_ef_construct=cfg["qdrant"]["hnsw_ef_construct"],
    )

    metrics["qdrant_collection"] = collection_name
    metrics["vector_dim"]        = int(active_vectors.shape[1])
    metrics["vectors_stored"]    = int(client.count(collection_name).count)

    # ── Smoke test ────────────────────────────────────────────────────────────
    smoke_test(client, collection_name, active_vectors[0])

    return metrics
