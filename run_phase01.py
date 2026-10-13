"""
Phase 01 Runner — Data Collection
==================================
Run this script to execute the full data collection pipeline:
  1. Scrape Futurepedia        → data/raw_futurepedia.csv
  2. Fetch ProductHunt API     → data/raw_producthunt.csv
  3. Fetch Hugging Face API    → data/raw_huggingface.csv
  4. Scrape AwesomeList (GH)   → data/raw_awesomelist.csv
  5. Scrape FutureTools.io     → data/raw_futuretools.csv
  6. Merge all five            → data/raw_tools.csv

Usage (from the toolsense/ directory):
    python run_phase01.py

Requirements:
    - .env file with PRODUCTHUNT_TOKEN set (copy from .env.example)
    - pip install -r requirements.txt

Re-scraping: delete the individual raw_*.csv files you want refreshed
  before running — the script skips sources with existing data files.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def check_env() -> bool:
    token = os.getenv("PRODUCTHUNT_TOKEN")
    if not token or token == "your_developer_token_here":
        log.error(
            "PRODUCTHUNT_TOKEN is not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Add your Developer Token from producthunt.com/v2/oauth/applications\n"
        )
        return False
    return True


def main():
    if not check_env():
        sys.exit(1)

    # Add project root to path
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    from src.data_pipeline.scraper_futurepedia import scrape_futurepedia
    from src.data_pipeline.scraper_producthunt import fetch_posts
    from src.data_pipeline.scraper_huggingface import fetch_spaces
    from src.data_pipeline.scraper_awesomelist import run as scrape_awesomelist
    from src.data_pipeline.scraper_futuretools import run as scrape_futuretools
    from src.data_pipeline.merger import merge

    import yaml
    with open(root / "config" / "base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    def has_data(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 200

    # ── Step 1: Scrape Futurepedia ───────────────────────────────────────────
    fp_path = root / "data" / "raw_futurepedia.csv"
    fp_path.parent.mkdir(exist_ok=True)

    if has_data(fp_path):
        log.info(f"STEP 1 — Futurepedia data already exists ({fp_path}), skipping.")
    else:
        log.info("=" * 60)
        log.info("STEP 1 — Scraping Futurepedia")
        log.info("=" * 60)
        try:
            fp_df = scrape_futurepedia(
                max_tools=cfg["scraping"].get("futurepedia_max_tools", 3000),
                page_delay=cfg["scraping"].get("futurepedia_delay_seconds", 1.0),
            )
            if len(fp_df) > 0:
                fp_df.to_csv(fp_path, index=False)
                log.info(f"Futurepedia: {len(fp_df)} tools saved to {fp_path}\n")
            else:
                log.warning("Futurepedia returned 0 tools.\n")
        except Exception as e:
            log.warning(f"Futurepedia scraper failed: {e}\nContinuing without it.\n")

    # ── Step 2: Fetch ProductHunt ─────────────────────────────────────────────
    ph_path = root / "data" / "raw_producthunt.csv"

    if has_data(ph_path):
        log.info(f"STEP 2 — ProductHunt data already exists ({ph_path}), skipping.")
    else:
        log.info("=" * 60)
        log.info("STEP 2 — Fetching ProductHunt API")
        log.info("=" * 60)
        token = os.getenv("PRODUCTHUNT_TOKEN")
        ph_df = fetch_posts(
            token=token,
            max_posts=cfg["scraping"]["producthunt_max_posts"],
            delay=cfg["scraping"]["producthunt_delay_seconds"],
        )
        if len(ph_df) > 0:
            ph_df.to_csv(ph_path, index=False)
            log.info(f"ProductHunt: {len(ph_df)} tools saved to {ph_path}\n")
        else:
            log.warning("ProductHunt returned 0 tools — not overwriting existing file.\n")

    # ── Step 3: Fetch Hugging Face Spaces ────────────────────────────────────
    hf_path = root / "data" / "raw_huggingface.csv"

    if has_data(hf_path):
        log.info(f"STEP 3 — Hugging Face data already exists ({hf_path}), skipping.")
    else:
        log.info("=" * 60)
        log.info("STEP 3 — Fetching Hugging Face Spaces (no auth required)")
        log.info("=" * 60)
        hf_df = fetch_spaces(
            max_spaces=cfg["scraping"].get("huggingface_max_spaces", 3000),
            delay=cfg["scraping"].get("huggingface_delay_seconds", 0.3),
            min_likes=cfg["scraping"].get("huggingface_min_likes", 0),
        )
        if len(hf_df) > 0:
            hf_df.to_csv(hf_path, index=False)
            log.info(f"HuggingFace: {len(hf_df)} spaces saved to {hf_path}\n")
        else:
            log.warning("HuggingFace returned 0 spaces — not overwriting existing file.\n")

    # ── Step 4: Scrape Awesome AI Tools (GitHub) ──────────────────────────────
    al_path = root / "data" / "raw_awesomelist.csv"

    if has_data(al_path):
        log.info(f"STEP 4 — AwesomeList data already exists ({al_path}), skipping.")
    else:
        log.info("=" * 60)
        log.info("STEP 4 — Scraping Awesome AI Tools (GitHub markdown)")
        log.info("=" * 60)
        try:
            al_df = scrape_awesomelist()
            if len(al_df) > 0:
                log.info(f"AwesomeList: {len(al_df)} tools saved to {al_path}\n")
            else:
                log.warning("AwesomeList returned 0 tools.\n")
        except Exception as e:
            log.warning(f"AwesomeList scraper failed: {e}\nContinuing without it.\n")

    # ── Step 5: Scrape FutureTools.io ─────────────────────────────────────────
    ft_path = root / "data" / "raw_futuretools.csv"

    if has_data(ft_path):
        log.info(f"STEP 5 — FutureTools data already exists ({ft_path}), skipping.")
    else:
        log.info("=" * 60)
        log.info("STEP 5 — Scraping FutureTools.io")
        log.info("=" * 60)
        try:
            ft_df = scrape_futuretools()
            if len(ft_df) > 0:
                log.info(f"FutureTools: {len(ft_df)} tools saved to {ft_path}\n")
            else:
                log.warning("FutureTools returned 0 tools.\n")
        except Exception as e:
            log.warning(f"FutureTools scraper failed: {e}\nContinuing without it.\n")

    # ── Step 6: Merge all five ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 6 — Merging all datasets")
    log.info("=" * 60)

    fp_ok = fp_path.exists()  and fp_path.stat().st_size > 100
    ph_ok = ph_path.exists()  and ph_path.stat().st_size > 100
    hf_ok = hf_path.exists()  and hf_path.stat().st_size > 100
    al_ok = al_path.exists()  and al_path.stat().st_size > 100
    ft_ok = ft_path.exists()  and ft_path.stat().st_size > 100

    if not any([fp_ok, ph_ok, hf_ok, al_ok, ft_ok]):
        log.error("All scrapers produced no data. Cannot merge.")
        sys.exit(1)

    for name, ok in [
        ("Futurepedia", fp_ok), ("ProductHunt", ph_ok),
        ("HuggingFace", hf_ok), ("AwesomeList", al_ok),
        ("FutureTools", ft_ok),
    ]:
        if not ok:
            log.warning(f"{name} data missing — will be skipped in merge.")

    merged_df = merge(
        taaft_path=fp_path,
        ph_path=ph_path,
        hf_path=hf_path,
        awesomelist_path=al_path,
        futuretools_path=ft_path,
        out_path=root / "data" / "raw_tools.csv",
    )

    log.info("\n" + "=" * 60)
    log.info("PHASE 01 COMPLETE")
    log.info(f"  Total tools collected: {len(merged_df)}")
    log.info(f"  Output: data/raw_tools.csv")
    log.info("=" * 60)
    log.info("Next step: run Phase 02 (Data Quality).")


if __name__ == "__main__":
    main()
