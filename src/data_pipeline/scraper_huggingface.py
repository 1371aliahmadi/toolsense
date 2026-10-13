"""
Hugging Face Spaces Scraper
============================
Fetches AI demo spaces from the Hugging Face Hub public API.

No API key or authentication required.
Endpoint: https://huggingface.co/api/spaces

The API returns metadata for public Spaces (interactive ML demos).
Each Space has: id, cardData (title, tags, description from README),
likes, sdk (gradio/streamlit), and pipeline_tag (task type).

We map pipeline_tag → category and sdk → pricing_tier marker.

Docs: https://huggingface.co/docs/hub/api#get-apispaces

Usage:
    python -m src.data_pipeline.scraper_huggingface
"""

import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HuggingFace] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

HF_API_URL = "https://huggingface.co/api/spaces"

# pipeline_tag values from HF → human-readable category for ToolSense
PIPELINE_TO_CATEGORY: dict[str, str] = {
    "text-generation": "Text Generation",
    "text-to-image": "Image Generation",
    "image-to-text": "Image Understanding",
    "text-classification": "Text Classification",
    "translation": "Translation",
    "summarization": "Summarization",
    "question-answering": "Question Answering",
    "conversational": "Conversational AI",
    "text-to-speech": "Text to Speech",
    "automatic-speech-recognition": "Speech Recognition",
    "image-classification": "Image Classification",
    "object-detection": "Object Detection",
    "image-segmentation": "Image Segmentation",
    "video-classification": "Video Understanding",
    "feature-extraction": "Embeddings",
    "fill-mask": "Language Model",
    "token-classification": "NLP",
    "sentence-similarity": "Semantic Search",
    "zero-shot-classification": "Text Classification",
    "depth-estimation": "Computer Vision",
    "image-to-image": "Image Editing",
    "audio-classification": "Audio AI",
    "text-to-video": "Video Generation",
    "reinforcement-learning": "Reinforcement Learning",
    "robotics": "Robotics",
    "tabular-classification": "Data Science",
    "tabular-regression": "Data Science",
    "time-series-forecasting": "Forecasting",
}

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class HFSpace:
    tool_name: str = ""
    description: str = ""
    category: str = ""
    pricing_tier: str = "Free"   # All HF Spaces are free to use
    upvotes: int = 0
    user_count: int = 0
    url: str = ""
    source: str = "huggingface"


# ── Fetcher ───────────────────────────────────────────────────────────────────

def fetch_spaces(
    max_spaces: int = 1000,
    delay: float = 0.3,
    page_size: int = 100,
    min_likes: int = 0,
) -> pd.DataFrame:
    """
    Fetch AI Spaces from the Hugging Face Hub public API.

    Strategy:
    - Use page-based pagination (?p=0, ?p=1, ...) with limit=page_size
    - Sort by likes (most popular first) so we get the best tools first
    - Accept any space that has a title + at least one metadata signal
      (pipeline_tag, sdk, or tags) — generate a description from those
      when no explicit description exists
    - Stop early after 3 consecutive pages with 0 useful tools

    Args:
        max_spaces: stop after collecting this many spaces
        delay:      seconds between API requests (be polite)
        page_size:  items per request (max 100 for HF /api/spaces)
        min_likes:  minimum likes threshold (0 = no filter)

    Returns:
        DataFrame with unified schema columns
    """
    spaces: list[HFSpace] = []
    seen_ids: set[str] = set()
    consecutive_empty = 0
    MAX_CONSECUTIVE_EMPTY = 3
    page_num = 0

    # First request — explicit params
    next_url: str | None = HF_API_URL
    next_params: dict | None = {
        "limit": page_size,
        "sort": "likes",
        "direction": -1,
        "full": "true",
    }

    while next_url and len(spaces) < max_spaces:
        log.info(f"Fetching page {page_num} (collected: {len(spaces)}/{max_spaces})")

        try:
            resp = requests.get(next_url, params=next_params, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning(f"Request failed on page {page_num}: {e}. Stopping.")
            break

        # ── Follow Link header for next page (HF cursor pagination) ──────────
        link_header = resp.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        if match:
            next_url = match.group(1)
            next_params = None   # URL already contains all params
        else:
            next_url = None      # no more pages

        items = resp.json()

        if not items or not isinstance(items, list):
            log.info("Empty response — reached end of available spaces.")
            break

        added_this_page = 0

        for item in items:
            space_id = item.get("id", "")   # e.g. "stabilityai/stable-diffusion"
            if not space_id or space_id in seen_ids:
                continue
            seen_ids.add(space_id)

            likes = item.get("likes", 0) or 0
            if min_likes > 0 and likes < min_likes:
                continue

            # ── Extract tool name ─────────────────────────────────────────────
            card_data = item.get("cardData") or {}
            pipeline_tag = item.get("pipeline_tag", "") or ""
            sdk = item.get("sdk", "") or ""
            tags = card_data.get("tags") or []

            title = card_data.get("title", "")
            if not title:
                # Fall back: repo-name part of "owner/repo-name", humanised
                slug = space_id.split("/")[-1] if "/" in space_id else space_id
                title = slug.replace("-", " ").replace("_", " ").title()
            if not title:
                continue

            # ── Extract description ───────────────────────────────────────────
            description = card_data.get("short_description", "")
            if not description:
                description = card_data.get("description", "")

            # If still no description, synthesise one from available metadata
            if not description or len(description.split()) < 3:
                parts: list[str] = []
                if pipeline_tag:
                    readable = PIPELINE_TO_CATEGORY.get(
                        pipeline_tag, pipeline_tag.replace("-", " ").title()
                    )
                    parts.append(readable)
                elif tags:
                    clean_tags = [t.replace("-", " ") for t in tags[:2] if len(t) > 2]
                    if clean_tags:
                        parts.append(", ".join(clean_tags))
                if sdk:
                    parts.append(f"interactive demo built with {sdk.capitalize()}")
                if parts:
                    description = f"{title} — {'; '.join(parts)}."
                else:
                    # Truly nothing to say — skip
                    continue

            # Trim very long descriptions to ~200 chars
            if len(description) > 200:
                description = description[:197].rsplit(" ", 1)[0] + "…"

            # Final quality gate: must have at least 4 words
            if len(description.split()) < 4:
                continue

            # ── Category ──────────────────────────────────────────────────────
            category = PIPELINE_TO_CATEGORY.get(pipeline_tag, "")
            if not category:
                category = tags[0].replace("-", " ").title() if tags else "AI Tool"

            # ── Build record ──────────────────────────────────────────────────
            space = HFSpace(
                tool_name=title,
                description=description,
                category=category,
                pricing_tier="Free",
                upvotes=likes,
                user_count=0,
                url=f"https://huggingface.co/spaces/{space_id}",
                source="huggingface",
            )
            spaces.append(space)
            added_this_page += 1

            if len(spaces) >= max_spaces:
                break

        log.info(f"  → {added_this_page} added this page (total: {len(spaces)})")

        if added_this_page == 0:
            consecutive_empty += 1
            log.info(f"  (no useful tools — {consecutive_empty}/{MAX_CONSECUTIVE_EMPTY} empty pages)")
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                log.info("3 consecutive empty pages — stopping early.")
                break
        else:
            consecutive_empty = 0

        if not next_url:
            log.info("No next-page link — reached end of API results.")
            break

        page_num += 1
        time.sleep(delay)

    df = pd.DataFrame([asdict(s) for s in spaces])
    df = df.drop_duplicates(subset=["tool_name"])
    log.info(f"Fetched {len(df)} unique spaces from Hugging Face")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_spaces(max_spaces=1000, delay=0.3, min_likes=5)
    out_path = Path("data/raw_huggingface.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Saved to {out_path}")
    print(df.head(10).to_string())
    print(f"\nCategory distribution:\n{df['category'].value_counts().head(15).to_string()}")
