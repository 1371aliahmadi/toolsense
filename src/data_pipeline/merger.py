"""
Data Merger
===========
Merges TAAFT, ProductHunt, and Hugging Face Spaces datasets into a single
unified CSV.

Strategy:
  1. Normalise tool_name in all datasets (lowercase, strip punctuation)
  2. Join on normalised name — matched tools get TAAFT/HF description +
     ProductHunt popularity signals
  3. Unmatched tools from any source are kept with partial data
  4. Output: data/raw_tools.csv with unified schema

Unified schema:
  tool_name | description | category | pricing_tier |
  upvotes   | user_count  | source   | url

Usage:
    python -m src.data_pipeline.merger
"""

import re
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Merger] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Unified schema ────────────────────────────────────────────────────────────

UNIFIED_COLUMNS = [
    "tool_name",
    "description",
    "category",
    "pricing_tier",
    "upvotes",
    "user_count",
    "source",
    "url",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

# ── Merge logic ───────────────────────────────────────────────────────────────

def _load_csv_safe(path: Path, name: str) -> pd.DataFrame:
    """Load a CSV, returning empty DataFrame (not raising) if missing or empty."""
    if not path.exists():
        log.warning(f"{name} CSV not found at {path} — skipping.")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    try:
        df = pd.read_csv(path)
        if df.empty:
            log.warning(f"{name} CSV is empty — skipping.")
            return pd.DataFrame(columns=UNIFIED_COLUMNS)
        log.info(f"  → {len(df)} rows from {name}")
        return df
    except pd.errors.EmptyDataError:
        log.warning(f"{name} CSV is empty — skipping.")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)


def merge(
    taaft_path: str | Path = "data/raw_taaft.csv",
    ph_path: str | Path = "data/raw_producthunt.csv",
    hf_path: str | Path = "data/raw_huggingface.csv",
    awesomelist_path: str | Path = "data/raw_awesomelist.csv",
    futuretools_path: str | Path = "data/raw_futuretools.csv",
    out_path: str | Path = "data/raw_tools.csv",
) -> pd.DataFrame:
    """
    Load all source CSVs, deduplicate, and return the unified DataFrame.

    Supports up to 5 sources: TAAFT/Futurepedia, ProductHunt, HuggingFace,
    AwesomeList (GitHub), and FutureTools.io.

    Merge strategy:
    - Primary key for deduplication: normalised tool_name
    - Preference order for description: TAAFT > HF > ProductHunt > extras
    - Popularity signals (upvotes, user_count): take max across matched sources
    - Source tag: joined source names if matched across sources
    """
    taaft_path       = Path(taaft_path)
    ph_path          = Path(ph_path)
    hf_path          = Path(hf_path)
    awesomelist_path = Path(awesomelist_path)
    futuretools_path = Path(futuretools_path)
    out_path         = Path(out_path)

    # ── Load all sources ──────────────────────────────────────────────────────
    log.info("Loading source datasets …")
    taaft       = _load_csv_safe(taaft_path,       "TAAFT/Futurepedia")
    ph          = _load_csv_safe(ph_path,           "ProductHunt")
    hf          = _load_csv_safe(hf_path,           "HuggingFace")
    awesomelist = _load_csv_safe(awesomelist_path,  "AwesomeList")
    futuretools = _load_csv_safe(futuretools_path,  "FutureTools")

    if taaft.empty and ph.empty and hf.empty and awesomelist.empty and futuretools.empty:
        log.error("All sources are empty. Cannot merge.")
        raise SystemExit(1)

    # ── Normalise names ───────────────────────────────────────────────────────
    for df in [taaft, ph, hf, awesomelist, futuretools]:
        if not df.empty and "tool_name" in df.columns:
            df["_norm"] = df["tool_name"].astype(str).apply(normalise_name)

    # ── Tag each source before combining ─────────────────────────────────────
    if not taaft.empty:
        taaft["source"] = taaft.get("source", pd.Series("taaft", index=taaft.index)).fillna("taaft")
    if not ph.empty:
        ph["source"] = ph.get("source", pd.Series("producthunt", index=ph.index)).fillna("producthunt")
    if not hf.empty:
        hf["source"] = hf.get("source", pd.Series("huggingface", index=hf.index)).fillna("huggingface")
    if not awesomelist.empty:
        awesomelist["source"] = awesomelist.get("source", pd.Series("awesomelist", index=awesomelist.index)).fillna("awesomelist")
    if not futuretools.empty:
        futuretools["source"] = futuretools.get("source", pd.Series("futuretools", index=futuretools.index)).fillna("futuretools")

    # ── Ensure all frames have unified columns ────────────────────────────────
    all_cols = UNIFIED_COLUMNS + ["_norm"]
    for df in [taaft, ph, hf, awesomelist, futuretools]:
        for col in all_cols:
            if col not in df.columns:
                df[col] = None

    # ── Concatenate all sources ───────────────────────────────────────────────
    log.info("Concatenating all sources …")
    frames = [df[all_cols] for df in [taaft, ph, hf, awesomelist, futuretools] if not df.empty]
    combined = pd.concat(frames, ignore_index=True)

    log.info(f"  → {len(combined)} total rows before deduplication")

    # ── Deduplicate on normalised name ────────────────────────────────────────
    # For duplicates: prefer row with longer description, then higher upvotes.
    # Source tag: join all source names that contributed.
    combined["upvotes"] = pd.to_numeric(combined["upvotes"], errors="coerce").fillna(0).astype(int)
    combined["user_count"] = pd.to_numeric(combined["user_count"], errors="coerce").fillna(0).astype(int)
    combined["desc_len"] = combined["description"].fillna("").str.len()

    # Sort so best row (longest description, most upvotes) comes first
    combined = combined.sort_values(
        ["_norm", "desc_len", "upvotes"],
        ascending=[True, False, False],
    )

    # Aggregate source tags before dedup
    source_map = (
        combined.groupby("_norm")["source"]
        .apply(lambda s: "+".join(sorted(set(s.dropna()))))
        .to_dict()
    )
    combined["source"] = combined["_norm"].map(source_map)

    # Also aggregate upvotes (take max across sources for same tool)
    upvotes_map = combined.groupby("_norm")["upvotes"].max().to_dict()
    combined["upvotes"] = combined["_norm"].map(upvotes_map)

    # Keep first (best) row per normalised name
    combined = combined.drop_duplicates(subset=["_norm"], keep="first")

    log.info(f"  → {len(combined)} unique tools after deduplication")

    # ── Final cleanup ─────────────────────────────────────────────────────────
    combined["tool_name"]    = combined["tool_name"].astype(str).str.strip()
    combined["description"]  = combined["description"].fillna("").str.strip()
    combined["category"]     = combined["category"].fillna("").str.strip()
    combined["pricing_tier"] = combined["pricing_tier"].fillna("Unknown").str.strip()
    combined["upvotes"]      = combined["upvotes"].fillna(0).astype(int)
    combined["user_count"]   = combined["user_count"].fillna(0).astype(int)
    combined["url"]          = combined["url"].fillna("").str.strip()

    combined = combined[UNIFIED_COLUMNS].reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    log.info(f"\nUnified dataset: {len(combined)} tools → {out_path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    log.info("\n── Source breakdown ──")
    log.info(combined["source"].value_counts().to_string())

    log.info("\n── Pricing tier distribution ──")
    log.info(combined["pricing_tier"].value_counts().head(10).to_string())

    return combined

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = merge(
        taaft_path="data/raw_futurepedia.csv",
        ph_path="data/raw_producthunt.csv",
        hf_path="data/raw_huggingface.csv",
        awesomelist_path="data/raw_awesomelist.csv",
        futuretools_path="data/raw_futuretools.csv",
    )
    print(f"\nFinal shape: {df.shape}")
    print(df.head(5).to_string())
