"""
Evaluation Metrics — Phase 05 / 06
=====================================
Implements Precision@K and Recall@K for the ToolSense recommender.

Ground truth format:
  A list of (query, relevant_set) pairs where relevant_set is a set of
  strings that should appear in the top-K results' tool_name or category.

Phase 05 uses a small hardcoded proxy test set.
Phase 08 replaces this with Label Studio annotations.
"""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ── Proxy test set ────────────────────────────────────────────────────────────
# Each entry: (query, set of keywords that should appear in results)
# A result is "relevant" if any keyword appears in tool_name or category (case-insensitive)

PROXY_TEST_SET = [
    ("generate AI-powered videos for YouTube",          {"video", "youtube"}),
    ("write better marketing copy and emails",          {"email", "marketing", "writing"}),
    ("customer support chatbot for my website",         {"chat", "support", "messaging", "assistant"}),
    ("generate images from text prompts",               {"image", "design", "photo"}),
    ("automate coding and fix bugs",                    {"developer", "code", "engineering"}),
    ("transcribe audio and video to text",              {"transcri", "audio", "speech"}),
    ("summarize long documents and PDFs",               {"summar", "document", "writing"}),
    ("SEO optimization for my website",                 {"seo", "marketing"}),
    ("create social media posts automatically",         {"social", "marketing", "content"}),
    ("data analysis and business intelligence",         {"data", "analytics", "business"}),
    ("AI tool for graphic design and logos",            {"design", "graphic", "image"}),
    ("scheduling and calendar management",              {"productivity", "calendar", "schedule"}),
    ("generate podcast from text",                      {"audio", "podcast", "voice"}),
    ("sales outreach and cold email automation",        {"sales", "email", "outreach"}),
    ("code review and debugging assistant",             {"developer", "code", "engineering"}),
    ("HR recruiting and candidate screening",           {"hr", "recruit", "hiring"}),
    ("translate text between languages",                {"translat", "language"}),
    ("create presentations and slide decks",            {"presentation", "design", "slides"}),
    ("monitor social media mentions",                   {"social", "marketing", "monitoring"}),
    ("AI writing assistant for blog posts",             {"writing", "content", "blog"}),
]


# ── Metric functions ──────────────────────────────────────────────────────────

def is_relevant(result: dict, keywords: set[str]) -> bool:
    """
    A result is relevant if any keyword (substring) appears in the
    tool_name or category fields (case-insensitive).
    """
    haystack = (
        result.get("tool_name", "").lower() + " " +
        result.get("category", "").lower() + " " +
        result.get("description", "").lower()[:200]
    )
    return any(kw.lower() in haystack for kw in keywords)


def precision_at_k(results: list[dict], keywords: set[str], k: int) -> float:
    """Fraction of top-K results that are relevant."""
    if not results:
        return 0.0
    top_k = results[:k]
    hits = sum(1 for r in top_k if is_relevant(r, keywords))
    return hits / len(top_k)


@dataclass
class EvalResult:
    precision_at_5:  float
    precision_at_10: float
    mean_score:      float   # primary Optuna objective
    # NOTE: Recall is NOT computed here because it requires knowing the total
    # number of relevant items in the full catalogue — unknown without full
    # annotation. Recall is computed properly in Phase 08 (LLM annotation).


def evaluate(
    retrieve_fn,          # callable(query, **kwargs) -> list[dict]
    test_set: list = None,
    k_values: list[int] = None,
    **retrieve_kwargs,
) -> EvalResult:
    """
    Run retrieval on all test queries and compute average Precision@K / Recall@K.

    Args:
        retrieve_fn:      The ranker.retrieve function
        test_set:         List of (query, keywords_set) — defaults to PROXY_TEST_SET
        k_values:         [5, 10]
        retrieve_kwargs:  Passed to retrieve_fn (weights, qdrant_path, etc.)
    """
    if test_set is None:
        test_set = PROXY_TEST_SET
    if k_values is None:
        k_values = [5, 10]

    p5_scores, p10_scores = [], []

    for query, keywords in test_set:
        try:
            results = retrieve_fn(query=query, **retrieve_kwargs)
        except Exception as e:
            log.warning(f"Retrieval failed for '{query}': {e}")
            results = []

        p5_scores.append(precision_at_k(results, keywords, 5))
        p10_scores.append(precision_at_k(results, keywords, 10))

    p5  = sum(p5_scores)  / len(p5_scores)
    p10 = sum(p10_scores) / len(p10_scores)

    # Primary objective: average of P@5 and P@10
    mean = (p5 + p10) / 2

    return EvalResult(
        precision_at_5=round(p5, 4),
        precision_at_10=round(p10, 4),
        mean_score=round(mean, 4),
    )
