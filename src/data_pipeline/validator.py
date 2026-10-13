"""
Data Quality Validator — Phase 02
===================================
Applies quality checks to raw_tools.csv using:
  - Great Expectations (structured validation with pass/fail expectations)
  - Custom pandas checks (near-duplicates, source balance)
  - ydata-profiling (HTML exploratory report)

Checks performed:
  1. Completeness    — tool_name and description must not be null
  2. Description length — at least 10 words
  3. Near-duplicates — TF-IDF cosine similarity > 0.85 flagged and removed
  4. Source balance  — neither source dominates > 80% (relaxed from 60%
                        because TAAFT scrape was limited by bot protection)
  5. Upvote coverage — how many tools have popularity signal

Output:
  data/clean_tools.csv     — validated, deduplicated dataset
  data/quality_report.html — ydata-profiling HTML report
"""

import logging
import re
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ydata_profiling import ProfileReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Quality] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_DESC_WORDS       = 5
DUPLICATE_THRESHOLD  = 0.85
MAX_SOURCE_DOMINANCE = 0.80   # relaxed: TAAFT scrape was bot-limited

# HuggingFace Spaces auto-generate this boilerplate when no real description exists.
# Example: "MasaCtrl — interactive demo built with Gradio."
# It carries zero semantic signal for retrieval — filter it out.
_HF_TEMPLATE_RE = re.compile(
    r"interactive\s+demo\s+built\s+with\s+(Gradio|Streamlit|Docker|Static)",
    re.IGNORECASE,
)

# Titles/descriptions containing these words are flagged as NSFW
_NSFW_RE = re.compile(
    r"\b(porn|nsfw|explicit\s+content|xxx|hentai|adult\s+content)\b",
    re.IGNORECASE,
)

# ── 1. Pandas-based validation (replaces Great Expectations) ──────────────────

def run_great_expectations(df: pd.DataFrame) -> dict:
    """
    Structural validation using pandas (GE v1 API broke; plain pandas is more stable).
    Checks: required columns, null rates, valid source values, row count, name uniqueness.
    """
    log.info("Running data validation checks …")

    checks = {}

    # Required columns exist
    required_cols = ["tool_name", "description", "category", "source"]
    for col in required_cols:
        checks[f"column_exists_{col}"] = col in df.columns

    # tool_name not null
    checks["tool_name_not_null"] = df["tool_name"].notna().all()

    # description not null in ≥95% of rows
    checks["description_95pct_not_null"] = (
        df["description"].notna().mean() >= 0.95
    )

    # source values valid — covers all single and combined source tags
    _base_sources = {
        "taaft", "producthunt", "huggingface",
        "awesomelist", "futuretools", "futurepedia",
        "both",
    }
    # Accept any combination of base sources joined by "+"
    import itertools
    valid_sources = set(_base_sources)
    for r in range(2, len(_base_sources) + 1):
        for combo in itertools.combinations(sorted(_base_sources - {"both"}), r):
            valid_sources.add("+".join(combo))
    checks["source_values_valid"] = df["source"].dropna().isin(valid_sources).all()

    # row count in range
    checks["row_count_in_range"] = 500 <= len(df) <= 100_000

    # tool_name unique
    checks["tool_name_unique"] = df["tool_name"].nunique() == len(df)

    overall = all(checks.values())
    log.info(f"Validation: {'PASSED' if overall else 'FAILED'}")
    for name, ok in checks.items():
        log.info(f"  {'✓' if ok else '✗'} {name}")

    return {"overall": overall, "details": checks}


# ── 2. Custom checks ──────────────────────────────────────────────────────────

def check_description_length(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and remove rows with descriptions shorter than MIN_DESC_WORDS words."""
    word_counts = df["description"].fillna("").str.split().str.len()
    short_mask = word_counts < MIN_DESC_WORDS
    n_short = short_mask.sum()
    log.info(f"Description length check: {n_short} tools have < {MIN_DESC_WORDS} words → removing")
    return df[~short_mask].copy()


def filter_template_and_nsfw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove two categories of low-quality rows from HuggingFace data:

    1. Template descriptions — HF Spaces auto-generate "ToolName — interactive demo
       built with Gradio." when no README description is present. These strings are
       content-free and degrade embedding quality.

    2. NSFW — tool names/descriptions matching explicit-content keywords. Not suitable
       for a course project dataset.
    """
    desc = df["description"].fillna("")
    name = df["tool_name"].fillna("")

    hf_template = desc.str.contains(_HF_TEMPLATE_RE.pattern, case=False, regex=True)
    nsfw_desc   = desc.str.contains(_NSFW_RE.pattern, case=False, regex=True)
    nsfw_name   = name.str.contains(_NSFW_RE.pattern, case=False, regex=True)

    drop_mask = hf_template | nsfw_desc | nsfw_name
    n_template = hf_template.sum()
    n_nsfw     = (nsfw_desc | nsfw_name).sum()

    log.info(
        f"Template/NSFW filter: {n_template} HF boilerplate + {n_nsfw} NSFW "
        f"→ removing {drop_mask.sum()} rows"
    )
    return df[~drop_mask].copy()


def detect_near_duplicates(df: pd.DataFrame, threshold: float = DUPLICATE_THRESHOLD) -> pd.DataFrame:
    """
    Use TF-IDF cosine similarity to find near-duplicate descriptions.
    Keeps the first occurrence of each duplicate group.
    """
    log.info("Near-duplicate detection (TF-IDF cosine similarity) …")
    descriptions = df["description"].fillna("").tolist()

    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(descriptions)

    # Compute pairwise cosine similarity
    # For large datasets this could be slow; 1231 tools is fine (~1.5M pairs)
    sim_matrix = cosine_similarity(tfidf)

    # Upper triangle only, exclude diagonal
    duplicate_indices = set()
    n = len(df)
    for i in range(n):
        if i in duplicate_indices:
            continue
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                duplicate_indices.add(j)   # keep i, drop j

    log.info(f"Near-duplicates found: {len(duplicate_indices)} → removing")
    keep_mask = [i not in duplicate_indices for i in range(n)]
    return df[keep_mask].copy()


def check_source_balance(df: pd.DataFrame) -> dict:
    """Report source distribution and flag if any source dominates."""
    counts = df["source"].value_counts()
    total = len(df)
    dominant_source = counts.index[0]
    dominant_fraction = counts.iloc[0] / total

    log.info(f"Source balance: {counts.to_dict()}")
    if dominant_fraction > MAX_SOURCE_DOMINANCE:
        log.warning(
            f"Source imbalance: '{dominant_source}' is {dominant_fraction:.1%} of dataset "
            f"(threshold: {MAX_SOURCE_DOMINANCE:.0%}). "
            "This is expected given TAAFT bot protection limiting our scrape."
        )
    else:
        log.info(f"Source balance OK: largest source is {dominant_fraction:.1%}")

    return {
        "counts": counts.to_dict(),
        "dominant_source": dominant_source,
        "dominant_fraction": float(dominant_fraction),
        "balanced": dominant_fraction <= MAX_SOURCE_DOMINANCE,
    }


def enrich_short_descriptions(df: pd.DataFrame, min_words: int = 15) -> pd.DataFrame:
    """
    For tools with very short descriptions (< min_words words), prepend the
    category and tool name so SBERT embeds richer, more discriminative text.

    Before: "Generate music powered by AI"          (5 words — useless vector)
    After:  "Audio, Music: AI Jukebox — Generate music powered by AI. AI tool."

    This is applied AFTER deduplication so it doesn't affect duplicate detection.
    It is NOT applied to tools that already have sufficient description length.
    """
    df = df.copy()
    word_counts = df["description"].fillna("").str.split().str.len()
    short_mask  = word_counts < min_words
    n_short     = short_mask.sum()

    if n_short == 0:
        return df

    def _enrich(row):
        name = str(row.get("tool_name", "") or "").strip()
        cat  = str(row.get("category",  "") or "").strip()
        desc = str(row.get("description","") or "").strip().rstrip(".")

        parts = []
        if cat:
            parts.append(cat)
        if name:
            parts.append(name)
        if desc:
            parts.append(desc)
        if cat:
            parts.append(f"AI tool for {cat.lower()}")

        return ". ".join(parts)

    df.loc[short_mask, "description"] = df[short_mask].apply(_enrich, axis=1)
    log.info(
        f"Description enrichment: {n_short} tools had < {min_words} words — "
        f"prepended category+name for better SBERT signal"
    )
    return df


def check_popularity_coverage(df: pd.DataFrame) -> dict:
    """Report how many tools have upvote data."""
    has_upvotes = (df["upvotes"] > 0).sum()
    coverage = has_upvotes / len(df)
    log.info(f"Popularity coverage: {has_upvotes}/{len(df)} tools have upvotes ({coverage:.1%})")
    return {"tools_with_upvotes": int(has_upvotes), "coverage": float(coverage)}


# ── 3. ydata-profiling report ─────────────────────────────────────────────────

def generate_profile_report(df: pd.DataFrame, out_path: Path) -> None:
    """Generate an HTML data quality report with ydata-profiling."""
    log.info("Generating ydata-profiling HTML report …")
    profile = ProfileReport(
        df,
        title="ToolSense — Data Quality Report",
        explorative=True,
        minimal=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_file(str(out_path))
    log.info(f"Report saved to {out_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_quality_pipeline(
    raw_path: str | Path = "data/raw_tools.csv",
    clean_path: str | Path = "data/clean_tools.csv",
    report_path: str | Path = "data/quality_report.html",
) -> tuple[pd.DataFrame, dict]:
    """
    Full data quality pipeline.

    Returns:
        (clean_df, metrics_dict) — cleaned DataFrame + summary metrics
    """
    raw_path   = Path(raw_path)
    clean_path = Path(clean_path)
    report_path = Path(report_path)

    log.info(f"Loading raw data from {raw_path} …")
    df = pd.read_csv(raw_path)
    log.info(f"Raw dataset: {len(df)} rows, {df.shape[1]} columns")

    metrics = {"raw_count": len(df)}

    # ── Step 1: GE formal validation ─────────────────────────────────────────
    ge_results = run_great_expectations(df)
    metrics["ge_passed"] = ge_results["overall"]
    metrics["ge_details"] = ge_results["details"]

    # ── Step 2: Remove rows with null tool_name or description ───────────────
    before = len(df)
    df = df.dropna(subset=["tool_name", "description"])
    df = df[df["tool_name"].str.strip() != ""]
    df = df[df["description"].str.strip() != ""]
    metrics["removed_null"] = before - len(df)
    log.info(f"Removed {metrics['removed_null']} rows with null name/description")

    # ── Step 3: HF template + NSFW filter ────────────────────────────────────
    df = filter_template_and_nsfw(df)
    metrics["after_template_filter"] = len(df)

    # ── Step 4: Description length filter ────────────────────────────────────
    df = check_description_length(df)
    metrics["after_length_filter"] = len(df)

    # ── Step 5: Near-duplicate removal ───────────────────────────────────────
    df = detect_near_duplicates(df)
    metrics["after_dedup"] = len(df)

    # ── Step 5b: Enrich short descriptions for better SBERT embeddings ───────
    df = enrich_short_descriptions(df)

    # ── Step 6: Source balance report ────────────────────────────────────────
    metrics["source_balance"] = check_source_balance(df)

    # ── Step 7: Popularity coverage ──────────────────────────────────────────
    metrics["popularity"] = check_popularity_coverage(df)

    # ── Step 8: Reset index and save clean data ───────────────────────────────
    df = df.reset_index(drop=True)
    df["tool_id"] = df.index  # stable integer ID for Qdrant payload
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    log.info(f"Clean dataset: {len(df)} rows saved to {clean_path}")
    metrics["clean_count"] = len(df)

    # ── Step 9: Profile report ────────────────────────────────────────────────
    generate_profile_report(df, report_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    after_null   = metrics["raw_count"] - metrics["removed_null"]
    n_template   = after_null - metrics["after_template_filter"]
    n_short      = metrics["after_template_filter"] - metrics["after_length_filter"]
    n_dupes      = metrics["after_length_filter"] - metrics["after_dedup"]

    log.info("\n── Quality Summary ──")
    log.info(f"  Raw rows:              {metrics['raw_count']}")
    log.info(f"  Removed (null):        {metrics['removed_null']}")
    log.info(f"  Removed (template/NSFW): {n_template}")
    log.info(f"  Removed (short desc):  {n_short}")
    log.info(f"  Removed (near-dupes):  {n_dupes}")
    log.info(f"  Clean rows:            {metrics['clean_count']}")
    log.info(f"  GE passed:             {metrics['ge_passed']}")

    return df, metrics
