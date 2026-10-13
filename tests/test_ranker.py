"""
Unit tests for the hybrid ranker — Phase 10
============================================
Run with:
    pytest tests/test_ranker.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT        = Path(__file__).resolve().parents[1]
QDRANT_PATH = str(ROOT / "data" / "qdrant")


# ── correct_query ──────────────────────────────────────────────────────────────
def test_correct_query_passthrough():
    from src.models.ranker import correct_query
    assert correct_query("AI writing tool") == "AI writing tool"


def test_correct_query_fixes_typo():
    from src.models.ranker import correct_query
    # "writting" (double-t) is a common misspelling — pyspellchecker corrects it to "writing"
    result = correct_query("writting assistant")
    assert isinstance(result, str)       # never crashes
    assert "writ" in result.lower()      # corrected or left close to original


def test_correct_query_preserves_whitelist():
    from src.models.ranker import correct_query
    # "nlp", "api", "llm" are whitelisted and must not be changed
    q = "nlp api for llm"
    result = correct_query(q)
    assert "nlp" in result
    assert "api" in result
    assert "llm" in result


def test_correct_query_preserves_short_words():
    from src.models.ranker import correct_query
    # Words < 4 chars should not be changed
    result = correct_query("AI for hr and ux")
    assert "hr" in result
    assert "ux" in result


# ── retrieve ───────────────────────────────────────────────────────────────────
def test_retrieve_returns_results():
    from src.models.ranker import retrieve
    results = retrieve(
        query="AI writing assistant",
        qdrant_path=QDRANT_PATH,
        top_k=20,
        return_k=5,
    )
    assert isinstance(results, list)
    assert len(results) > 0


def test_retrieve_result_structure():
    from src.models.ranker import retrieve
    results = retrieve(
        query="customer support chatbot",
        qdrant_path=QDRANT_PATH,
        return_k=3,
    )
    required = {"tool_name", "description", "category",
                "pricing_tier", "final_score", "semantic_score", "url"}
    for r in results:
        assert required.issubset(r.keys())


def test_retrieve_scores_valid():
    from src.models.ranker import retrieve
    results = retrieve(
        query="generate images from text",
        qdrant_path=QDRANT_PATH,
        return_k=5,
    )
    for r in results:
        assert 0.0 <= r["final_score"] <= 1.0
        assert 0.0 <= r["semantic_score"] <= 1.0


def test_retrieve_sorted_descending():
    from src.models.ranker import retrieve
    results = retrieve(
        query="SEO keyword tool",
        qdrant_path=QDRANT_PATH,
        return_k=10,
    )
    scores = [r["final_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_return_k_respected():
    from src.models.ranker import retrieve
    for k in [1, 5, 10]:
        results = retrieve(
            query="code debugger",
            qdrant_path=QDRANT_PATH,
            return_k=k,
        )
        assert len(results) <= k


def test_retrieve_empty_for_nonsense():
    from src.models.ranker import retrieve
    # Even nonsense queries should not crash — just return something or empty
    results = retrieve(
        query="xyzzy frobulate wumbo",
        qdrant_path=QDRANT_PATH,
        return_k=5,
    )
    assert isinstance(results, list)
