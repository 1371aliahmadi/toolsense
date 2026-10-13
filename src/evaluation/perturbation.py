"""
Perturbation Analysis — Phase 07
==================================
Tests how robust the recommender is against noisy or imperfect queries.

Four perturbation types:
  1. typo          — random character swaps in query words (simulate typing errors)
  2. truncation    — keep only the first 40% of query words
  3. wrong_category — append a misleading category keyword
  4. synonym       — replace key verbs/nouns with synonyms

For each perturbation type we:
  - Apply it to all 20 proxy test queries
  - Run retrieval with best weights
  - Compute Precision@5 and NDCG@5
  - Compare against the clean baseline from Phase 06
  - Log degradation (delta) to W&B
"""

import random
import logging
from typing import Callable

log = logging.getLogger(__name__)

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)


# ── Synonym map ───────────────────────────────────────────────────────────────
# Maps common task-describing words to weaker/different alternatives.
SYNONYM_MAP = {
    "generate":     "make",
    "create":       "build",
    "write":        "produce",
    "automate":     "handle",
    "summarize":    "condense",
    "transcribe":   "convert",
    "optimize":     "improve",
    "schedule":     "plan",
    "monitor":      "watch",
    "translate":    "convert",
    "analyze":      "study",
    "analysis":     "study",
    "review":       "check",
    "design":       "make",
    "assistant":    "helper",
    "chatbot":      "bot",
    "code":         "script",
    "marketing":    "promotion",
    "email":        "message",
    "social":       "online",
    "content":      "material",
    "podcast":      "audio",
    "blog":         "article",
    "outreach":     "messaging",
    "recruiting":   "hiring",
    "presentations":"slides",
    "debugging":    "troubleshooting",
}

# ── Wrong-category keywords (intentionally misleading) ────────────────────────
MISLEADING_CATEGORIES = [
    "agriculture", "legal", "healthcare", "sports", "travel",
    "finance", "real estate", "manufacturing", "logistics",
]


# ── Perturbation functions ────────────────────────────────────────────────────

def perturb_typo(query: str, error_rate: float = 0.15) -> str:
    """
    Swap or drop random characters in roughly `error_rate` fraction of characters.
    Only modifies alphabetic characters mid-word.
    """
    chars = list(query)
    n_errors = max(1, int(len(chars) * error_rate))
    # Pick positions that are letters (not spaces or punctuation)
    letter_positions = [i for i, c in enumerate(chars) if c.isalpha() and i > 0]
    targets = random.sample(letter_positions, min(n_errors, len(letter_positions)))

    for pos in targets:
        action = random.choice(["swap", "drop", "replace"])
        if action == "swap" and pos + 1 < len(chars) and chars[pos + 1].isalpha():
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif action == "drop":
            chars[pos] = ""
        else:
            chars[pos] = random.choice("abcdefghijklmnopqrstuvwxyz")

    return "".join(chars)


def perturb_truncation(query: str, keep_fraction: float = 0.40) -> str:
    """Keep only the first N words of the query."""
    words = query.split()
    n_keep = max(1, int(len(words) * keep_fraction))
    return " ".join(words[:n_keep])


def perturb_wrong_category(query: str) -> str:
    """Append a misleading category keyword to the query."""
    misleading = random.choice(MISLEADING_CATEGORIES)
    return f"{query} for {misleading} industry"


def perturb_synonym(query: str) -> str:
    """Replace task-describing words with synonyms."""
    words = query.split()
    new_words = [SYNONYM_MAP.get(w.lower(), w) for w in words]
    return " ".join(new_words)


# ── Registry ──────────────────────────────────────────────────────────────────

PERTURBATION_TYPES: dict[str, Callable[[str], str]] = {
    "typo":           perturb_typo,
    "truncation":     perturb_truncation,
    "wrong_category": perturb_wrong_category,
    "synonym":        perturb_synonym,
}


def apply_perturbation(query: str, perturbation_type: str) -> str:
    """Apply a named perturbation to a query string."""
    fn = PERTURBATION_TYPES.get(perturbation_type)
    if fn is None:
        raise ValueError(f"Unknown perturbation: {perturbation_type}. "
                         f"Choose from {list(PERTURBATION_TYPES)}")
    return fn(query)
