"""
TAAFT Scraper — There's An AI For That (Selenium version)
==========================================================
Uses Selenium + headless Chrome to bypass bot detection on TAAFT.
webdriver-manager auto-downloads the correct ChromeDriver version.

Verified structure (2026-06-17):
  - Listing page: /ais/?page=N  (~20 tools per page)
  - Fields per card: name, tagline, category tag, pricing, url

Usage:
    python -m src.data_pipeline.scraper_taaft
"""

import time
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict

import pandas as pd
import yaml
from bs4 import BeautifulSoup

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TAAFT] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "base_config.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TAAFTTool:
    tool_name: str = ""
    description: str = ""
    category: str = ""
    pricing_tier: str = ""
    url: str = ""
    source: str = "taaft"

# ── Browser setup ─────────────────────────────────────────────────────────────

def _get_chrome_major_version() -> int | None:
    """
    Detect installed Chrome's major version on Windows/Mac/Linux.
    Returns e.g. 149 for Chrome 149.x.x.x, or None if undetectable.
    """
    import subprocess, re
    import os
    user = os.environ.get("USERPROFILE", "")
    candidates = [
        # Windows — system install
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        # Windows — user install (most common on personal machines)
        os.path.join(user, r"AppData\Local\Google\Chrome\Application\chrome.exe"),
        # Mac
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        # Linux
        "google-chrome", "google-chrome-stable", "chromium-browser", "chromium",
    ]
    for path in candidates:
        try:
            out = subprocess.check_output(
                [path, "--version"], stderr=subprocess.DEVNULL, timeout=5
            ).decode()
            m = re.search(r"(\d+)\.", out)
            if m:
                return int(m.group(1))
        except Exception:
            continue
    return None


def make_driver() -> uc.Chrome:
    """
    Create an undetected Chrome driver (visible window).
    Running without headless is required — TAAFT's Cloudflare protection
    detects and blocks headless Chrome. A visible Chrome window will open
    briefly while scraping; this is expected and normal.

    Automatically detects installed Chrome version and pins ChromeDriver
    to match — prevents the common 'ChromeDriver only supports Chrome N'
    version mismatch error.
    """
    options = uc.ChromeOptions()
    # NOTE: headless intentionally disabled — TAAFT blocks headless browsers
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--disable-notifications")

    # Pin driver version to match installed Chrome
    version = _get_chrome_major_version()
    if version:
        log.info(f"Detected Chrome version: {version} — pinning ChromeDriver to match")
        driver = uc.Chrome(options=options, use_subprocess=True, version_main=version)
    else:
        log.warning("Could not detect Chrome version — letting undetected_chromedriver auto-select")
        driver = uc.Chrome(options=options, use_subprocess=True)
    return driver

# ── Parser ────────────────────────────────────────────────────────────────────

def _slug_to_name(slug: str) -> str:
    """
    Convert a URL slug to a readable tool name as a last-resort fallback.
    'flowova-ai' → 'Flowova Ai'
    """
    return slug.replace("-", " ").title()


def _looks_like_number(text: str) -> bool:
    """Return True if text is a pure number or formatted count (e.g. '4,305' '14315.0')."""
    cleaned = re.sub(r"[\d,.\s]", "", text)
    return len(cleaned) == 0 and len(text.strip()) > 0


def parse_page_html(html: str) -> list[TAAFTTool]:
    """
    Parse the rendered HTML of one TAAFT listing page.

    Robust strategy (handles TAAFT DOM changes):
    1. Find all <a href="/ai/{slug}/"> anchors.
    2. For each unique slug, collect ALL matching anchors (there can be 2-3
       per card: the name link, a save-count link, a karma link).
    3. Pick the anchor whose text is NOT a pure number as the tool name.
       If all are numbers, fall back to slug-derived name.
    4. Walk up from the chosen anchor to find description, category, pricing.
    """
    soup = BeautifulSoup(html, "lxml")
    tools = []

    # Group all anchors by slug so we can pick the best name
    slug_anchors: dict[str, list] = {}
    for anchor in soup.find_all("a", href=re.compile(r"/ai/[^/]+/")):
        href = anchor.get("href", "")
        slug_match = re.search(r"/ai/([^/]+)/", href)
        if not slug_match:
            continue
        slug = slug_match.group(1)
        slug_anchors.setdefault(slug, []).append(anchor)

    for slug, anchors in slug_anchors.items():
        tool = TAAFTTool()

        # ── Pick best anchor: prefer one whose text is not a number ──────────
        chosen_anchor = None
        for anchor in anchors:
            text = anchor.get_text(strip=True)
            if text and not _looks_like_number(text):
                chosen_anchor = anchor
                tool.tool_name = text
                break

        if chosen_anchor is None:
            # All anchor texts were numbers — use slug as fallback name
            tool.tool_name = _slug_to_name(slug)
            chosen_anchor = anchors[0]

        href = chosen_anchor.get("href", "")
        tool.url = (
            href if href.startswith("http")
            else f"https://theresanaiforthat.com{href}"
        )

        # ── Walk up from chosen anchor to card container ──────────────────────
        card = chosen_anchor
        for _ in range(8):
            card = card.parent
            if card is None:
                break
            lines = [
                ln.strip()
                for ln in card.get_text(separator="\n", strip=True).splitlines()
                if ln.strip()
            ]
            for line in lines:
                if (
                    line != tool.tool_name
                    and len(line.split()) >= 4
                    and not line.startswith("http")
                    and not _looks_like_number(line)
                    and not re.match(r"^(Free|From|\$|Freemium|Open)", line, re.I)
                    and "Released" not in line
                    and "karma" not in line.lower()
                    and "saved" not in line.lower()
                ):
                    tool.description = line
                    break
            if tool.description:
                break

        # ── Category from /task/ link ─────────────────────────────────────────
        if card is not None:
            task_link = card.find("a", href=re.compile(r"/task/[^/]+/"))
            if task_link:
                tool.category = task_link.get_text(strip=True)

        # ── Pricing ───────────────────────────────────────────────────────────
        if card is not None:
            card_text = card.get_text(" ", strip=True)
            m = re.search(
                r"(Free\s*\+?\s*from\s*\$[\d.]+[/\w]*"
                r"|Free"
                r"|From\s*\$[\d.]+[/\w]*"
                r"|\$[\d.]+[/\w]*"
                r"|Freemium"
                r"|Open\s*Source)",
                card_text,
                re.IGNORECASE,
            )
            tool.pricing_tier = m.group(0).strip() if m else "Unknown"

        # Require at least a name and some description
        if tool.tool_name and tool.description:
            tools.append(tool)

    return tools

# ── Scraper ───────────────────────────────────────────────────────────────────

def scrape_taaft(max_pages: int = 60, delay: float = 2.0) -> pd.DataFrame:
    """
    Scrape TAAFT listing pages using undetected-chromedriver.
    Bypasses Cloudflare bot detection that blocks standard requests/Selenium.

    Args:
        max_pages: maximum pages to scrape (~20 tools each)
        delay:     seconds to wait between pages

    Returns:
        DataFrame with tool data
    """
    log.info("Starting undetected Chrome (headless) …")
    driver = make_driver()
    all_tools: list[TAAFTTool] = []

    try:
        # Warm up — visit homepage first
        log.info("Visiting homepage to establish session …")
        driver.get("https://theresanaiforthat.com/")
        time.sleep(3)

        for page in range(1, max_pages + 1):
            url = f"https://theresanaiforthat.com/ais/?page={page}"
            log.info(f"Scraping page {page}/{max_pages} — {url}")

            driver.get(url)

            # Wait for page to load, then log title for diagnostics
            time.sleep(3)
            log.info(f"  Page title: '{driver.title}'")

            # Wait until at least one /ai/ link appears in the DOM
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "a[href*='/ai/']")
                    )
                )
            except Exception:
                log.warning(
                    f"Page {page}: timed out. Title was: '{driver.title}'. "
                    "If this says 'Just a moment' it's a Cloudflare challenge. Stopping."
                )
                break

            # Extra wait for lazy-loaded content
            time.sleep(2)

            tools = parse_page_html(driver.page_source)

            if not tools:
                log.info(f"No tools on page {page}. Reached end.")
                break

            all_tools.extend(tools)
            log.info(f"  → {len(tools)} tools (total: {len(all_tools)})")
            time.sleep(delay)

    finally:
        driver.quit()
        log.info("Browser closed.")

    df = pd.DataFrame([asdict(t) for t in all_tools])
    df = df.drop_duplicates(subset=["tool_name"])
    log.info(f"Scraping complete: {len(df)} unique tools from TAAFT")
    return df

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    df = scrape_taaft(
        max_pages=cfg["scraping"]["taaft_max_pages"],
        delay=cfg["scraping"]["taaft_delay_seconds"],
    )
    out_path = Path("data/raw_taaft.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Saved to {out_path}")
    print(df.head(10).to_string())
