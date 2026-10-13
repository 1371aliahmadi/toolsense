"""
FutureTools.io Scraper
======================
Scrapes AI tool listings from futuretools.io.

FutureTools is a Webflow CMS site. Tool cards live under /tools
and are fully server-rendered HTML (no JS needed for initial load).

Strategy:
  1. Fetch /tools page (with ?page=N or Webflow offset params)
  2. Parse Webflow CMS item cards: tool name, description, tags, URL
  3. Map tags → ToolSense category
  4. Output CSV with unified schema (same as other scrapers)

NOTE: This scraper must be run on your machine — it needs real internet
      access to futuretools.io (GitHub CI/sandbox may be blocked).

Usage:
    python -m src.data_pipeline.scraper_futuretools
"""

import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FutureTools] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

BASE_URL  = "https://www.futuretools.io"
TOOLS_URL = "https://www.futuretools.io/tools"

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw_futuretools.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tag / category text → ToolSense category names
TAG_TO_CATEGORY: dict[str, str] = {
    "text":           "Writing, Text Generation",
    "writing":        "Writing, Text Generation",
    "copywriting":    "Writing, Copywriting",
    "code":           "Code, Developer Tools",
    "coding":         "Code, Developer Tools",
    "developer":      "Code, Developer Tools",
    "image":          "Image Generation, Design",
    "art":            "Image Generation, Design",
    "design":         "Image Generation, Design",
    "video":          "Video Generation",
    "audio":          "Audio, Music",
    "music":          "Audio, Music",
    "voice":          "Audio, Voice Cloning",
    "speech":         "Audio, Voice Cloning",
    "marketing":      "Marketing, SEO",
    "seo":            "Marketing, SEO",
    "social media":   "Marketing, Social Media",
    "productivity":   "Productivity",
    "research":       "Research, Academia",
    "chatbot":        "Conversational AI",
    "chat":           "Conversational AI",
    "customer support": "Customer Support",
    "customer service": "Customer Support",
    "hr":             "HR, Recruiting",
    "recruiting":     "HR, Recruiting",
    "data":           "Data Science, Analytics",
    "analytics":      "Data Science, Analytics",
    "finance":        "Finance, Fintech",
    "legal":          "Legal, Compliance",
    "education":      "Education, E-Learning",
    "summarizer":     "Summarization",
    "transcription":  "Transcription, Speech to Text",
    "translation":    "Translation",
    "presentations":  "Presentations",
    "slides":         "Presentations",
    "email":          "Email, Communication",
    "search":         "Search, Discovery",
    "avatar":         "Avatar, Virtual Humans",
    "3d":             "3D, Spatial AI",
    "gaming":         "Gaming",
    "healthcare":     "Healthcare",
}


@dataclass
class FutureTool:
    tool_name:    str  = ""
    description:  str  = ""
    category:     str  = ""
    pricing_tier: str  = "Unknown"
    upvotes:      int  = 0
    user_count:   int  = 0
    url:          str  = ""
    source:       str  = "futuretools"


def _get(url: str, session: requests.Session, delay: float = 1.0) -> BeautifulSoup | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 404:
            log.warning(f"404 at {url}")
            return None
        resp.raise_for_status()
        time.sleep(delay)
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as e:
        log.warning(f"Request failed for {url}: {e}")
        return None


def _tags_to_category(tags: list[str]) -> str:
    """Map a list of tag strings to the best ToolSense category."""
    for tag in tags:
        t_lower = tag.lower().strip()
        for key, cat in TAG_TO_CATEGORY.items():
            if key in t_lower:
                return cat
    return tags[0].strip().title() if tags else "AI Tool"


def _parse_pricing(text: str) -> str:
    """Extract pricing tier from card text."""
    t = text.lower()
    if "open source" in t: return "Open Source"
    if "free trial" in t:  return "Free Trial"
    if "freemium"   in t:  return "Freemium"
    if "paid"       in t:  return "Paid"
    if "free"       in t:  return "Free"
    return "Unknown"


def parse_tools_page(soup: BeautifulSoup) -> list[FutureTool]:
    """
    Parse tool cards from a FutureTools listing page.

    Webflow CMS sites wrap each item in a div with class 'w-dyn-item'.
    Inside each item there's typically:
      - An <a> link to the tool page
      - A heading (h2/h3) for the tool name
      - A paragraph for description
      - Tag links or a category label
      - A pricing badge

    We try several selector strategies to be resilient to site changes.
    """
    tools: list[FutureTool] = []
    seen: set[str] = set()

    # Strategy 1: Webflow CMS pattern — look for w-dyn-item divs
    cards = soup.select("div.w-dyn-item")

    # Strategy 2: Fall back to any div/article that contains a link to /tools/
    if not cards:
        cards = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if re.match(r"^/tools/[^/]+/?$", href) or \
               re.match(r"^https://www\.futuretools\.io/tools/[^/]+/?$", href):
                # Walk up to find the card container
                parent = a.parent
                for _ in range(5):
                    if parent is None:
                        break
                    if parent.name in ("div", "article", "li", "section"):
                        cards.append(parent)
                        break
                    parent = parent.parent

    log.info(f"  Found {len(cards)} candidate cards on page")

    for card in cards:
        card_text = card.get_text(" ", strip=True)
        if len(card_text) < 20:
            continue

        # ── Tool URL ────────────────────────────────────────────────────────────
        tool_url = ""
        for a in card.find_all("a", href=True):
            href = a.get("href", "")
            if re.search(r"/tools/[^/]+/?$", href):
                tool_url = href if href.startswith("http") else urljoin(BASE_URL, href)
                break

        if not tool_url:
            continue

        # ── Tool name ───────────────────────────────────────────────────────────
        name = ""
        for tag in ["h2", "h3", "h4", "h1"]:
            el = card.find(tag)
            if el:
                name = el.get_text(strip=True)
                break
        if not name:
            # fallback: first <a> with /tools/ href text
            for a in card.find_all("a", href=True):
                if re.search(r"/tools/", a.get("href", "")):
                    candidate = a.get_text(strip=True)
                    if candidate and len(candidate) > 1:
                        name = candidate
                        break
        if not name or len(name) < 2:
            continue

        norm = name.lower().strip()
        if norm in seen:
            continue
        seen.add(norm)

        # ── Description ─────────────────────────────────────────────────────────
        description = ""
        for p in card.find_all("p"):
            txt = p.get_text(strip=True)
            if (txt and len(txt.split()) >= 4
                    and txt.lower() != name.lower()
                    and not txt.lower().startswith(("free", "paid", "freemium", "open"))):
                description = txt[:300]
                break

        if not description:
            # fallback: lines from card text
            lines = [ln.strip() for ln in card.get_text("\n", strip=True).splitlines()]
            for line in lines:
                if (len(line.split()) >= 4
                        and line.lower() != name.lower()
                        and not re.match(r"^(Free|Paid|Freemium|Open|Contact|\d+)", line, re.I)
                        and not line.startswith("http")):
                    description = line[:300]
                    break

        if not description:
            continue

        # ── Category / tags ─────────────────────────────────────────────────────
        tag_texts: list[str] = []
        for a in card.find_all("a", href=True):
            href = a.get("href", "")
            # Tag links often point to /tools?tags=... or /tag/... or /category/...
            if re.search(r"(tags?=|/tag/|/categor)", href, re.I):
                txt = a.get_text(strip=True)
                if txt and len(txt) > 1 and txt.lower() not in {"visit", "save"}:
                    tag_texts.append(txt)

        category = _tags_to_category(tag_texts) if tag_texts else "AI Tool"

        # ── Pricing ─────────────────────────────────────────────────────────────
        pricing = _parse_pricing(card_text)

        tools.append(FutureTool(
            tool_name=name,
            description=description,
            category=category,
            pricing_tier=pricing,
            url=tool_url,
        ))

    return tools


def scrape_futuretools(max_tools: int = 2000, delay: float = 1.0) -> pd.DataFrame:
    """
    Scrape AI tools from FutureTools.io.

    Tries pagination via ?page=N query parameter (common Webflow pattern).
    Stops when a page returns no new tools or we hit max_tools.

    Args:
        max_tools: Maximum number of tools to collect
        delay:     Seconds between requests (be polite)

    Returns:
        DataFrame with unified ToolSense schema columns
    """
    session = requests.Session()
    all_tools: list[FutureTool] = []
    seen_names: set[str] = set()
    consecutive_empty = 0
    MAX_CONSECUTIVE_EMPTY = 3

    log.info(f"Starting FutureTools scraper (max={max_tools})")

    for page in range(1, 200):
        if len(all_tools) >= max_tools:
            log.info(f"Reached max_tools={max_tools}, stopping.")
            break

        # Webflow pagination: ?page=N (1-indexed)
        url = TOOLS_URL if page == 1 else f"{TOOLS_URL}?page={page}"
        log.info(f"Fetching page {page}: {url}")

        soup = _get(url, session, delay=delay)
        if soup is None:
            log.warning(f"Failed to fetch page {page}, stopping.")
            break

        tools = parse_tools_page(soup)
        new_tools = []
        for t in tools:
            norm = t.tool_name.lower().strip()
            if norm not in seen_names:
                seen_names.add(norm)
                new_tools.append(t)

        log.info(f"  Page {page}: {len(new_tools)} new tools (total: {len(all_tools) + len(new_tools)})")
        all_tools.extend(new_tools)

        if len(new_tools) == 0:
            consecutive_empty += 1
            log.info(f"  No new tools ({consecutive_empty}/{MAX_CONSECUTIVE_EMPTY} empty pages)")
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                log.info("3 consecutive empty pages — stopping.")
                break
        else:
            consecutive_empty = 0

        # Check if there's a next-page link
        next_link = soup.find("a", string=re.compile(r"next|›|»", re.I))
        if not next_link and page > 1 and len(new_tools) == 0:
            log.info("No next-page link and no new tools — done.")
            break

    df = pd.DataFrame([asdict(t) for t in all_tools])
    if df.empty:
        log.warning("FutureTools scraper returned 0 tools.")
        return df

    df = df.drop_duplicates(subset=["tool_name"])
    log.info(f"\nFutureTools scraping complete: {len(df)} unique tools")

    from collections import Counter
    cats = Counter(df["category"].tolist())
    for cat, n in cats.most_common(10):
        log.info(f"  {cat}: {n}")

    return df


def run():
    df = scrape_futuretools(max_tools=2000, delay=1.0)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    log.info(f"Saved → {OUT_PATH}")
    return df


if __name__ == "__main__":
    df = run()
    print(f"\nTotal tools scraped: {len(df)}")
    if not df.empty:
        print(df[["tool_name", "category", "pricing_tier", "url"]].head(10).to_string())
        print(f"\nPricing breakdown:\n{df['pricing_tier'].value_counts().to_string()}")
        print(f"\nCategory breakdown:\n{df['category'].value_counts().head(15).to_string()}")
