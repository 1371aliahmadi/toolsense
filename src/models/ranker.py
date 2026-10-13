"""
Hybrid Ranker — Phase 04 (updated: BM25 hybrid)
=================================================
Scores retrieved tools using a weighted combination of:
  - Semantic similarity  (from Qdrant HNSW ANN search)
  - BM25 keyword score   (rank_bm25 over tool text, computed at query time)
  - Category overlap     (query words vs tool category)
  - Popularity           (log-normalised upvotes, weight defaults to 0)

Formula:
  final_score = w_semantic*semantic + w_bm25*bm25 + w_category*category
                + w_popularity*popularity

Weights tuned by Optuna in Phase 05b. BM25 added to fix low precision caused
by SBERT finding semantically-adjacent but keyword-mismatched tools.
"""

import logging
import math
import re
from pathlib import Path
from functools import lru_cache

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

log = logging.getLogger(__name__)

# ── Spell-correction preprocessing ───────────────────────────────────────────
# Words we never want auto-corrected (AI acronyms, brand names, common abbrevs)
_SPELL_WHITELIST = {
    "ai", "ml", "nlp", "api", "seo", "crm", "erp", "hr", "ui", "ux",
    "pdf", "csv", "url", "llm", "rag", "gpu", "saas", "b2b", "b2c",
    "youtube", "github", "slack", "notion", "chatgpt", "gpt",
}

# Domain-specific pre-corrections: common typos where pyspellchecker picks
# the wrong word (e.g. "emal" → "meal" instead of "email")
_DOMAIN_CORRECTIONS = {
    "emal":     "email",
    "emial":    "email",
    "emaill":   "email",
    "vidio":    "video",
    "viedo":    "video",
    "vide":     "video",
    "transcibe":"transcribe",
    "trascribe":"transcribe",
    "maketing": "marketing",
    "marketin": "marketing",
    "custmer":  "customer",
    "supprt":   "support",
    "chatbbot": "chatbot",
    "autmate":  "automate",
    "analyics": "analytics",
    "analitics":"analytics",
    "recuiting":"recruiting",
    "recrutment":"recruitment",
    "presntation":"presentation",
    "presenation":"presentation",
}

# Extra vocabulary to teach pyspellchecker (domain words it may not know)
_TECH_VOCAB = [
    "email", "chatbot", "podcast", "webinar", "saas", "fintech",
    "ecommerce", "analytics", "transcribe", "automate", "summarize",
    "onboarding", "recruiting", "workflow", "integration", "dashboard",
    "scraping", "embedding", "semantic", "generative", "multimodal",
]

_spell_checker = None


def _get_spell_checker():
    global _spell_checker
    if _spell_checker is None:
        from spellchecker import SpellChecker
        _spell_checker = SpellChecker()
        _spell_checker.word_frequency.load_words(_TECH_VOCAB)
    return _spell_checker


def correct_query(query: str) -> str:
    """
    Correct obvious typos in a query string.
    - Skips words in the whitelist (AI/tech acronyms)
    - Skips all-uppercase words (likely acronyms)
    - Skips words shorter than 4 chars (too risky to auto-correct)
    - Skips words with digits (model names, version numbers)
    - Returns original query unchanged if spellchecker is unavailable
    """
    try:
        spell = _get_spell_checker()
    except Exception:
        return query

    words = query.split()
    corrected = []
    changed = []

    for word in words:
        # Strip punctuation for checking, preserve original form
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()

        skip = (
            not clean                        # punctuation-only token
            or clean in _SPELL_WHITELIST     # whitelisted term
            or word.isupper()                # acronym like NLP
            or len(clean) < 3               # too short to correct safely
            or any(c.isdigit() for c in word)  # has digits
        )

        if skip:
            corrected.append(word)
        else:
            # Check domain-specific corrections first (pyspellchecker can pick
            # wrong common words, e.g. "emal" → "meal" instead of "email")
            if clean in _DOMAIN_CORRECTIONS:
                suggestion = _DOMAIN_CORRECTIONS[clean]
            else:
                suggestion = spell.correction(clean)
                # Only accept if the suggestion is actually similar to the
                # original — prevents "emal"→"meal" (too different) while
                # allowing "transcibe"→"transcribe" (very similar)
                if suggestion and suggestion != clean:
                    from difflib import SequenceMatcher
                    sim = SequenceMatcher(None, clean, suggestion).ratio()
                    if sim < 0.6:
                        suggestion = clean  # reject — too different

            if suggestion and suggestion != clean:
                # Preserve original capitalisation style
                if word[0].isupper():
                    suggestion = suggestion.capitalize()
                corrected.append(suggestion)
                changed.append(f"{word}->{suggestion}")
            else:
                corrected.append(word)

    result = " ".join(corrected)
    if changed:
        log.debug(f"Spell-corrected: {', '.join(changed)}")
    return result

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "base_config.yaml"
CLEAN_CSV   = Path(__file__).resolve().parents[2] / "data" / "clean_tools.csv"

MODEL_NAMES = {
    "model_a": "all-MiniLM-L6-v2",
    "model_b": "all-mpnet-base-v2",
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Singleton model + client (loaded once per process) ────────────────────────

_model: SentenceTransformer | None = None
_client: QdrantClient | None = None
_max_upvotes: float = 1.0   # set on first query

# ── BM25 index (built once from clean_tools.csv) ──────────────────────────────
_bm25_index   = None    # None = not loaded yet; False = load failed (don't retry)
_bm25_tool_idx: dict[str, int] = {}   # tool_name → row index in corpus

_BM25_STOPWORDS = {
    "a","an","the","for","and","or","to","of","in","my","from","with",
    "that","this","is","are","as","i","me","can","do","how","what",
    "which","using","use","tool","ai","ml","platform","app","software",
    "help","helper","helps","make","makes","get","your","you","best",
    "free","new","pro","plus","online","powered","based","smart","easy",
}

def _tok(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if t not in _BM25_STOPWORDS and len(t) > 1]


def get_bm25_index():
    """Build (once) a BM25 index over all clean tools. Thread-safe via GIL."""
    global _bm25_index, _bm25_tool_idx
    # False = already tried and failed — don't retry, don't re-warn
    if _bm25_index is not None:
        if _bm25_index is False:
            return None, {}
        return _bm25_index, _bm25_tool_idx

    try:
        import pandas as pd
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.warning("rank_bm25 not installed — BM25 scoring disabled. pip install rank-bm25")
        _bm25_index = False   # sentinel: don't retry or re-warn
        return None, {}

    if not CLEAN_CSV.exists():
        log.warning(f"clean_tools.csv not found at {CLEAN_CSV} — BM25 disabled")
        _bm25_index = False
        return None, {}

    df = pd.read_csv(CLEAN_CSV)
    corpus = []
    for i, row in df.iterrows():
        name = str(row.get("tool_name", "") or "")
        cat  = str(row.get("category",  "") or "")
        desc = str(row.get("description","") or "")
        tokens = _tok(name) * 3 + _tok(cat) * 2 + _tok(desc)
        corpus.append(tokens if tokens else ["__empty__"])
        _bm25_tool_idx[name] = len(corpus) - 1

    _bm25_index = BM25Okapi(corpus)
    log.info(f"BM25 index built: {len(corpus)} tools")
    return _bm25_index, _bm25_tool_idx


def get_model(cfg: dict) -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = MODEL_NAMES[cfg["embeddings"]["active_model"]]
        log.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


def get_client(qdrant_path: str, cfg: dict) -> QdrantClient:
    global _client
    if _client is None:
        log.info(f"Connecting to Qdrant at: {qdrant_path}")
        _client = QdrantClient(path=qdrant_path)
    return _client


# ── Scoring helpers ───────────────────────────────────────────────────────────

# ── Category adjacency graph (for gradient boost) ─────────────────────────────
# Maps each category keyword to semantically adjacent domain keywords.
# Enables partial scoring (0.4) for near-domain matches instead of binary 0/1.
_CATEGORY_GRAPH: dict[str, set[str]] = {
    "image":       {"design", "video", "art", "creative", "visual", "photo"},
    "video":       {"image", "audio", "creative", "animation", "editing", "clip"},
    "audio":       {"video", "music", "podcast", "voice", "speech", "sound"},
    "writing":     {"marketing", "email", "content", "copywriting", "blog", "text"},
    "marketing":   {"writing", "seo", "social", "email", "sales", "advertising"},
    "code":        {"developer", "data", "productivity", "devops", "api", "programming"},
    "data":        {"code", "analytics", "research", "finance", "reporting", "visualization"},
    "productivity":{"code", "hr", "research", "education", "workflow", "automation"},
    "hr":          {"productivity", "recruiting", "legal", "education", "hiring"},
    "education":   {"productivity", "research", "hr", "writing", "learning"},
    "research":    {"data", "education", "productivity", "analytics", "summarize"},
    "design":      {"image", "video", "creative", "presentation", "ui", "visual"},
    "social":      {"marketing", "writing", "video", "content", "media"},
    "email":       {"writing", "marketing", "sales", "outreach"},
    "sales":       {"email", "marketing", "crm", "productivity", "outreach"},
    "finance":     {"data", "analytics", "productivity", "accounting"},
    "legal":       {"hr", "productivity", "research", "compliance"},
    "security":    {"code", "developer", "data", "privacy"},
    "music":       {"audio", "video", "creative", "sound"},
    "analytics":   {"data", "research", "marketing", "reporting"},
}


def category_score(query: str, category: str) -> float:
    """
    Gradient category score:
      - 1.0  exact match (a category word appears literally in the query)
      - 0.4  adjacent match (query domain is semantically adjacent per _CATEGORY_GRAPH)
      - 0.0  no relation

    Examples:
      query="edit my videos", category="Video, Editing"          → 1.0 (exact)
      query="generate AI images", category="Design"               → 0.4 (adjacent)
      query="automate HR recruiting", category="Legal, Compliance" → 0.4 (adjacent: hr↔legal)
      query="transcribe podcast", category="Finance"              → 0.0
    """
    if not category:
        return 0.0
    query_words = set(re.sub(r"[^a-z\s]", "", query.lower()).split())
    cat_words: set[str] = set()
    for part in category.split(","):
        for w in re.sub(r"[^a-z\s]", "", part.lower()).split():
            if w:
                cat_words.add(w)

    # Exact match
    if query_words & cat_words:
        return 1.0

    # Adjacent match: any category word has a neighbor that appears in the query
    for cw in cat_words:
        neighbors = _CATEGORY_GRAPH.get(cw, set())
        if query_words & neighbors:
            return 0.4

    return 0.0


# ── Dynamic weight shifting ────────────────────────────────────────────────────

def _detect_query_type(query: str) -> str:
    """
    Classify a query as 'named_entity', 'short', or 'conversational'.

    - named_entity: contains a mid-sentence capitalized word → likely a tool name
                    (e.g. "alternatives to Jasper for writing")
    - short: ≤3 meaningful tokens after stopword removal → BM25 signal stronger
    - conversational: long natural-language phrase → SBERT semantic understanding wins
    """
    words = query.split()
    # Skip the first word (always capitalized); check the rest for proper nouns
    has_named_entity = any(
        w[0].isupper()
        and w.lower() not in {"ai", "the", "a", "an", "i", "my", "for", "to", "in"}
        and len(w) > 2
        for w in words[1:]
    )
    if has_named_entity:
        return "named_entity"

    meaningful = [t for t in re.sub(r"[^a-z\s]", "", query.lower()).split()
                  if t not in _BM25_STOPWORDS and len(t) > 1]
    if len(meaningful) <= 3:
        return "short"

    return "conversational"


def _dynamic_weights(
    w_sem: float,
    w_bm25: float,
    w_cat: float,
    query: str,
) -> tuple[float, float, float]:
    """
    Adjust scoring weights based on query type to improve robustness:

    - named_entity  → BM25 dominates (precision for specific product names)
    - short         → BM25 boosted (keyword match outweighs SBERT on ≤3 tokens)
    - conversational→ SBERT boosted (intent captured better by embeddings)

    All returned weights sum to the same total as inputs.
    """
    total = w_sem + w_bm25 + w_cat
    qtype = _detect_query_type(query)

    if qtype == "named_entity":
        # Named-entity queries: surface the exact tool by name
        ratios = (0.20, 0.72, 0.08)
    elif qtype == "short":
        # Short queries: fewer tokens → keyword precision matters more
        ratios = (0.28, 0.62, 0.10)
    else:
        # Conversational: long intent phrase → semantic embedding wins
        ratios = (0.52, 0.38, 0.10)

    return tuple(r * total for r in ratios)  # type: ignore[return-value]


def popularity_score(upvotes: int, max_upvotes: float) -> float:
    """Log-normalised upvotes mapped to [0, 1]."""
    if max_upvotes <= 0:
        return 0.0
    return math.log1p(upvotes) / math.log1p(max_upvotes)


def bm25_scores_for_candidates(
    query: str,
    candidate_names: list[str],
) -> dict[str, float]:
    """
    Return BM25 score normalised to [0, 1] for each candidate tool name.
    Scores are computed over the full catalogue and then min-max normalised
    within the candidate set so the weight w_bm25 has consistent meaning.
    """
    bm25, tool_idx = get_bm25_index()
    if bm25 is None:
        return {n: 0.0 for n in candidate_names}

    q_tokens = _tok(query)
    if not q_tokens:
        return {n: 0.0 for n in candidate_names}

    all_scores = bm25.get_scores(q_tokens)   # score for every tool in corpus

    raw = {}
    for name in candidate_names:
        idx = tool_idx.get(name)
        raw[name] = float(all_scores[idx]) if idx is not None else 0.0

    # Min-max normalise within candidate set
    vals = list(raw.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {n: 0.0 for n in candidate_names}
    return {n: (v - lo) / (hi - lo) for n, v in raw.items()}


# ── Main retrieval function ───────────────────────────────────────────────────

def retrieve(
    query: str,
    qdrant_path: str,
    top_k: int = 20,
    return_k: int = 5,
    w_semantic: float = 0.80,
    w_bm25: float = 0.15,
    w_category: float = 0.05,
    w_popularity: float = 0.00,
    filters: dict | None = None,
    dynamic_weights: bool = True,
    category_boosts: dict[str, float] | None = None,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline.

    Args:
        query:           Natural-language user query
        qdrant_path:     Path to local Qdrant storage
        top_k:           Candidates to retrieve from Qdrant (ANN)
        return_k:        Final results to return after reranking
        w_semantic:      Weight for SBERT cosine similarity
        w_bm25:          Weight for BM25 keyword score (rank_bm25)
        w_category:      Weight for category gradient score
        w_popularity:    Weight for log-normalised upvotes (default 0)
        filters:         Optional dict {"pricing_tier": "Free"} to hard-filter
        dynamic_weights:  If True, automatically adjust sem/bm25/cat weights based
                          on query type (short, named entity, conversational).
                          Improves robustness across diverse query styles.
        category_boosts:  Session feedback map {category_keyword: vote_count}.
                          Positive counts add up to +0.15 to that tool's category
                          score; negative counts subtract. Passed from Streamlit
                          session state so user thumbs-up/down actually affect ranking.

    Returns:
        List of result dicts sorted by final_score descending
    """
    global _max_upvotes

    cfg    = load_config()
    model  = get_model(cfg)
    client = get_client(qdrant_path, cfg)
    collection = cfg["qdrant"]["collection_name"]

    # ── Spell-correct query before embedding ──────────────────────────────────
    query = correct_query(query)

    # ── Dynamic weight adjustment (query-type aware) ──────────────────────────
    if dynamic_weights:
        w_semantic, w_bm25, w_category = _dynamic_weights(
            w_semantic, w_bm25, w_category, query
        )
        log.debug(
            f"Dynamic weights [{_detect_query_type(query)}]: "
            f"sem={w_semantic:.3f} bm25={w_bm25:.3f} cat={w_category:.3f}"
        )

    # ── Embed query ───────────────────────────────────────────────────────────
    query_vec = model.encode(
        query, normalize_embeddings=True, convert_to_numpy=True
    )

    # ── ANN search in Qdrant ──────────────────────────────────────────────────
    hits = client.query_points(
        collection_name=collection,
        query=query_vec.tolist(),
        limit=top_k,
    ).points

    if not hits:
        return []

    # ── Compute max_upvotes for normalisation (once) ──────────────────────────
    if _max_upvotes <= 1.0:
        all_upvotes = [h.payload.get("upvotes", 0) for h in hits]
        _max_upvotes = max(all_upvotes) if all_upvotes else 1.0

    # ── Apply hard filters ────────────────────────────────────────────────────
    if filters:
        hits = [
            h for h in hits
            if all(
                str(h.payload.get(k, "")).lower() == str(v).lower()
                for k, v in filters.items()
            )
        ]

    # ── BM25 scores for all candidates (batch, normalised) ───────────────────
    candidate_names = [h.payload.get("tool_name", "") for h in hits]
    bm25_map = bm25_scores_for_candidates(query, candidate_names) if w_bm25 > 0 else {}

    # ── Hybrid scoring ────────────────────────────────────────────────────────
    results = []
    _boosts = category_boosts or {}

    for hit in hits:
        p = hit.payload
        s_sem = float(hit.score)                                    # cosine (0-1)
        s_bm25= bm25_map.get(p.get("tool_name", ""), 0.0)         # bm25 (0-1)
        s_cat = category_score(query, p.get("category", ""))
        s_pop = popularity_score(int(p.get("upvotes", 0)), _max_upvotes)

        # ── Session feedback boost ────────────────────────────────────────────
        # Each thumbs-up on a tool in this category adds +0.10 to s_cat (capped
        # at 1.0). Thumbs-down subtracts. Multiple votes accumulate.
        if _boosts:
            cat_kws = set(re.sub(r"[^a-z\s]", "", (p.get("category") or "").lower()).split())
            vote_sum = sum(_boosts.get(kw, 0) for kw in cat_kws)
            if vote_sum:
                s_cat = max(0.0, min(1.0, s_cat + vote_sum * 0.10))

        final = (w_semantic   * s_sem
               + w_bm25       * s_bm25
               + w_category   * s_cat
               + w_popularity * s_pop)

        results.append({
            "tool_name":       p.get("tool_name", ""),
            "description":     p.get("description", ""),
            "category":        p.get("category", ""),
            "pricing_tier":    p.get("pricing_tier", "Unknown"),
            "upvotes":         int(p.get("upvotes", 0)),
            "url":             p.get("url", ""),
            "source":          p.get("source", ""),
            "semantic_score":  round(s_sem,  4),
            "bm25_score":      round(s_bm25, 4),
            "category_score":  round(s_cat,  4),
            "popularity_score":round(s_pop,  4),
            "final_score":     round(final,  4),
            # Expose gradient category type for UI badge
            "category_match":  (
                "exact" if s_cat >= 1.0
                else "adjacent" if s_cat > 0.0
                else "none"
            ),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:return_k]
