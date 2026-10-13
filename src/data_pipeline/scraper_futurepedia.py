"""
Futurepedia Scraper
===================
Scrapes AI tool listings from futurepedia.io using plain requests + BeautifulSoup.
No Selenium needed — pages are fully server-rendered HTML (Next.js SSR).

Strategy:
  1. Fetch /ai-tools to discover all parent category URLs
  2. Fetch each parent category page to discover subcategory URLs
  3. For each subcategory, paginate (?page=1, ?page=2, ...) until empty
  4. Parse tool cards: name, description, pricing, save-count, tags, URL

Why no Selenium: Unlike TAAFT (which uses Cloudflare + client-side rendering),
Futurepedia serves full HTML from the server. Requests with a browser-like
User-Agent is sufficient.

Usage:
    python -m src.data_pipeline.scraper_futurepedia
"""

import time
import logging
import re
from pathlib import Path
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Futurepedia] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

BASE_URL = "https://www.futurepedia.io"

# Mimic a real browser to avoid 403 responses
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Parent category slugs from /ai-tools main page
PARENT_CATEGORIES = [
    "productivity",
    "video",
    "text-generators",
    "business",
    "image",
    "automations",
    "art",
    "audio-generators",
    "code",
    "misc-tools",
]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class FPTool:
    tool_name: str = ""
    description: str = ""
    category: str = ""
    pricing_tier: str = "Unknown"
    upvotes: int = 0        # Futurepedia calls this "saves/bookmarks"
    user_count: int = 0
    url: str = ""
    source: str = "futurepedia"


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(url: str, session: requests.Session, delay: float = 1.0) -> BeautifulSoup | None:
    """Fetch a URL and return parsed BeautifulSoup, or None on failure."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        time.sleep(delay)
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as e:
        log.warning(f"Request failed for {url}: {e}")
        return None


# ── Subcategory discovery ─────────────────────────────────────────────────────

def _normalise_href(href: str) -> str:
    """Strip domain from absolute Futurepedia URLs → relative path."""
    for prefix in ("https://www.futurepedia.io", "http://www.futurepedia.io"):
        if href.startswith(prefix):
            return href[len(prefix):]
    return href


def get_subcategory_urls(session: requests.Session) -> list[tuple[str, str]]:
    """
    Fetch /ai-tools/{parent} pages for each parent category and collect
    all subcategory URLs.

    Returns list of (subcategory_slug, category_name) tuples.

    Note: Futurepedia uses ABSOLUTE hrefs (https://www.futurepedia.io/ai-tools/...)
    so we normalise them before matching.
    """
    subcategories: list[tuple[str, str]] = []
    seen: set[str] = set()

    for parent in PARENT_CATEGORIES:
        url = f"{BASE_URL}/ai-tools/{parent}"
        log.info(f"Discovering subcategories in: {parent}")
        soup = _get(url, session)
        if soup is None:
            continue

        for a in soup.find_all("a", href=True):
            href = _normalise_href(a.get("href", ""))
            # Match exactly /ai-tools/{slug} — no further slashes
            if not re.match(r"^/ai-tools/[^/]+/?$", href):
                continue
            slug = href.strip("/").split("/")[-1]
            if slug in seen or slug in PARENT_CATEGORIES or slug in ("", "ai-tools"):
                continue
            seen.add(slug)
            name = a.get_text(strip=True)
            name = re.sub(r"\s*\(\d+\)$", "", name).strip()
            if name and len(name) > 1:
                subcategories.append((slug, name))
                log.info(f"  Found subcategory: {slug} ({name})")

    log.info(f"Total subcategories discovered: {len(subcategories)}")
    return subcategories


# ── Tool card parser ──────────────────────────────────────────────────────────

def parse_tools_from_page(soup: BeautifulSoup, category: str) -> list[FPTool]:
    """
    Parse all tool cards from a rendered Futurepedia listing page.

    Card structure in the HTML (simplified):
      <a href="/tool/{slug}">ToolName</a>
      {PricingTier}{SaveCount}Add bookmark
      {Description text}
      <a href="/ai-tools/{tag}">#tag</a> ...
      <a href="https://...">Visit</a>
    """
    tools: list[FPTool] = []
    seen_slugs: set[str] = set()

    # Find all anchor tags pointing to /tool/{slug} — handles both relative and absolute hrefs
    for anchor in soup.find_all("a", href=True):
        href = _normalise_href(anchor.get("href", ""))
        if not re.match(r"^/tool/[^/]+/?$", href):
            continue
        slug = href.strip("/").split("/")[-1]
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        tool_name = anchor.get_text(strip=True)
        if not tool_name or len(tool_name) < 2:
            continue

        tool = FPTool(
            tool_name=tool_name,
            category=category,
            url=f"{BASE_URL}/tool/{slug}",
            source="futurepedia",
        )

        # ── Walk up to find the card container ───────────────────────────────
        card = anchor.parent
        for _ in range(8):
            if card is None:
                break
            text = card.get_text(" ", strip=True)
            # Card is big enough when it contains both a description and "Add bookmark"
            if "Add bookmark" in text and len(text) > 60:
                break
            card = card.parent

        if card is None:
            tools.append(tool)
            continue

        card_text = card.get_text(" ", strip=True)

        # ── Pricing tier ──────────────────────────────────────────────────────
        pricing_match = re.search(
            r"\b(Free Trial|Freemium|Free|Paid|Open Source|Contact for Pricing)\b",
            card_text,
            re.IGNORECASE,
        )
        if pricing_match:
            tool.pricing_tier = pricing_match.group(0).strip()

        # ── Save count (bookmark count = proxy for popularity) ────────────────
        # Pattern: "Freemium1247Add bookmark" → 1247
        save_match = re.search(
            r"(?:Free Trial|Freemium|Free|Paid|Open Source|Contact for Pricing)"
            r"\s*(\d+)\s*Add bookmark",
            card_text,
            re.IGNORECASE,
        )
        if save_match:
            tool.upvotes = int(save_match.group(1))

        # ── Description ───────────────────────────────────────────────────────
        # Description is the first sentence-like text that is NOT the tool name,
        # NOT a pricing string, NOT a tag, and has >= 4 words
        lines = [ln.strip() for ln in card.get_text("\n", strip=True).splitlines() if ln.strip()]
        for line in lines:
            if (
                line == tool_name
                or line.lower() in {"add bookmark", "editor's pick", "visit"}
                or re.match(r"^(Free|Paid|Freemium|Open|Contact|Rated)", line, re.I)
                or re.match(r"^#", line)
                or re.match(r"^\d+$", line)
                or len(line.split()) < 4
                or line.startswith("http")
            ):
                continue
            tool.description = line[:250]
            break

        if tool.description:
            tools.append(tool)

    return tools


# ── Category scraper ──────────────────────────────────────────────────────────

def scrape_subcategory(
    slug: str,
    category_name: str,
    session: requests.Session,
    max_pages: int = 20,
    delay: float = 1.0,
) -> list[FPTool]:
    """Scrape all pages of one subcategory."""
    all_tools: list[FPTool] = []
    consecutive_empty = 0

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/ai-tools/{slug}" + (f"?page={page}" if page > 1 else "")
        soup = _get(url, session, delay=delay)

        if soup is None:
            log.info(f"    [{slug}] page {page} → 404/error, stopping")
            break

        tools = parse_tools_from_page(soup, category_name)
        if not tools:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
        else:
            consecutive_empty = 0
            all_tools.extend(tools)
            log.info(f"    [{slug}] page {page} → {len(tools)} tools (total: {len(all_tools)})")

    return all_tools


# ── Main scraper ──────────────────────────────────────────────────────────────

def scrape_futurepedia(
    max_tools: int = 3000,
    page_delay: float = 1.0,
) -> pd.DataFrame:
    """
    Scrape AI tools from Futurepedia.

    Args:
        max_tools:   stop after collecting this many tools
        page_delay:  seconds between page requests (be polite)

    Returns:
        DataFrame with unified schema columns
    """
    session = requests.Session()
    all_tools: list[FPTool] = []
    seen_names: set[str] = set()

    # Step 1: Discover all subcategory URLs
    subcategories = get_subcategory_urls(session)

    if not subcategories:
        log.error("No subcategories found — check if Futurepedia structure changed.")
        return pd.DataFrame()

    # Step 2: Scrape each subcategory
    for slug, category_name in subcategories:
        if len(all_tools) >= max_tools:
            log.info(f"Reached max_tools={max_tools}, stopping.")
            break

        log.info(f"Scraping subcategory: {category_name} ({slug})")
        tools = scrape_subcategory(slug, category_name, session, delay=page_delay)

        # Deduplicate by tool name across categories
        for tool in tools:
            norm = tool.tool_name.lower().strip()
            if norm not in seen_names:
                seen_names.add(norm)
                all_tools.append(tool)

        log.info(f"  → {len(tools)} scraped, {len(all_tools)} total unique")

    df = pd.DataFrame([asdict(t) for t in all_tools])
    df = df.drop_duplicates(subset=["tool_name"])
    log.info(f"Futurepedia scraping complete: {len(df)} unique tools")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = scrape_futurepedia(max_tools=3000, page_delay=1.0)
    out_path = Path("data/raw_futurepedia.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} tools to {out_path}")
    print(df.head(10).to_string())
    print(f"\nPricing breakdown:\n{df['pricing_tier'].value_counts().to_string()}")
    print(f"\nCategory breakdown:\n{df['category'].value_counts().head(20).to_string()}")
