"""
Unit tests for the FastAPI backend — Phase 10
=============================================
Run with:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    return TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────────
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── /weights ───────────────────────────────────────────────────────────────────
def test_weights(client):
    resp = client.get("/weights")
    assert resp.status_code == 200
    data = resp.json()
    assert "w_semantic" in data
    assert "w_category" in data
    assert "w_popularity" in data
    # Weights should be floats between 0 and 1
    for v in data.values():
        assert 0.0 <= v <= 1.0


# ── /search ────────────────────────────────────────────────────────────────────
def test_search_basic(client):
    resp = client.post("/search", json={"query": "AI writing assistant", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "AI writing assistant"
    assert "results" in data
    assert "took_ms" in data
    assert len(data["results"]) <= 5


def test_search_returns_correct_fields(client):
    resp = client.post("/search", json={"query": "video generation tool", "top_k": 3})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) > 0
    required_fields = {"tool_name", "category", "description",
                       "pricing_tier", "final_score", "semantic_score"}
    for r in results:
        assert required_fields.issubset(r.keys())


def test_search_scores_in_range(client):
    resp = client.post("/search", json={"query": "automate customer support", "top_k": 5})
    assert resp.status_code == 200
    for r in resp.json()["results"]:
        assert 0.0 <= r["final_score"] <= 1.0
        assert 0.0 <= r["semantic_score"] <= 1.0


def test_search_results_sorted_by_score(client):
    resp = client.post("/search", json={"query": "code review AI", "top_k": 10})
    assert resp.status_code == 200
    scores = [r["final_score"] for r in resp.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_query_rejected(client):
    resp = client.post("/search", json={"query": "", "top_k": 5})
    assert resp.status_code == 422   # Pydantic validation error


def test_search_top_k_respected(client):
    for k in [1, 5, 10]:
        resp = client.post("/search", json={"query": "SEO tool", "top_k": k})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) <= k


def test_search_pricing_filter(client):
    resp = client.post("/search", json={
        "query": "marketing automation",
        "top_k": 10,
        "pricing_tier": "Free",
    })
    assert resp.status_code == 200
    for r in resp.json()["results"]:
        assert r["pricing_tier"].lower() == "free"
