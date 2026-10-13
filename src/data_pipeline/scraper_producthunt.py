"""
ProductHunt Scraper
===================
Fetches AI-tagged product launches via the ProductHunt GraphQL API.
Requires a Developer Token from:
  producthunt.com/v2/oauth/applications → Developer Tokens → Create Token

Docs: https://api.producthunt.com/v2/docs

Usage:
    python -m src.data_pipeline.scraper_producthunt
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

import requests
import pandas as pd
import yaml
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ProductHunt] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "base_config.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# ── GraphQL query ─────────────────────────────────────────────────────────────

POSTS_QUERY = """
query GetAIPosts($first: Int!, $after: String) {
  posts(
    first: $first,
    after: $after,
    topic: "artificial-intelligence",
    order: VOTES
  ) {
    edges {
      node {
        id
        name
        tagline
        description
        votesCount
        website
        createdAt
        reviewsCount
        topics {
          edges {
            node {
              name
            }
          }
        }
      }
      cursor
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PHProduct:
    tool_name: str = ""
    description: str = ""
    category: str = ""
    pricing_tier: str = "Unknown"  # PH doesn't expose pricing
    upvotes: int = 0
    user_count: int = 0
    url: str = ""
    source: str = "producthunt"

# ── Fetcher ───────────────────────────────────────────────────────────────────

def fetch_posts(
    token: str,
    max_posts: int = 2000,
    delay: float = 0.5,
    page_size: int = 20,
) -> pd.DataFrame:
    """
    Fetch AI-tagged posts from ProductHunt GraphQL API with cursor pagination.

    Args:
        token:     Developer token from ProductHunt
        max_posts: stop after collecting this many posts
        delay:     seconds between API calls
        page_size: posts per request (max 20 per PH docs)

    Returns:
        DataFrame with columns matching the unified schema
    """
    endpoint = "https://api.producthunt.com/v2/api/graphql"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    products: list[PHProduct] = []
    cursor: str | None = None
    page = 0

    while len(products) < max_posts:
        page += 1
        variables = {"first": page_size, "after": cursor}

        log.info(f"Fetching page {page} (collected: {len(products)}/{max_posts})")

        try:
            resp = requests.post(
                endpoint,
                json={"query": POSTS_QUERY, "variables": variables},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning(f"Request failed on page {page}: {e}. Stopping.")
            break

        data = resp.json()

        if "errors" in data:
            log.error(f"GraphQL errors: {data['errors']}")
            break

        posts_data = data.get("data", {}).get("posts", {})
        edges = posts_data.get("edges", [])
        page_info = posts_data.get("pageInfo", {})

        if not edges:
            log.info("No more posts returned. Done.")
            break

        for edge in edges:
            node = edge.get("node", {})

            # Topics → join as comma-separated category string
            topic_edges = node.get("topics", {}).get("edges", [])
            topics = [t["node"]["name"] for t in topic_edges if t.get("node")]
            category = ", ".join(topics) if topics else "AI"

            # Prefer longer description over tagline
            desc = node.get("description") or node.get("tagline") or ""

            p = PHProduct(
                tool_name=node.get("name", "").strip(),
                description=desc.strip(),
                category=category,
                upvotes=node.get("votesCount", 0) or 0,
                user_count=node.get("reviewsCount", 0) or 0,
                url=node.get("website", "") or "",
            )

            if p.tool_name and p.description:
                products.append(p)

        # Pagination
        if not page_info.get("hasNextPage"):
            log.info("Reached last page.")
            break

        cursor = page_info.get("endCursor")
        time.sleep(delay)

    df = pd.DataFrame([asdict(p) for p in products])
    df = df.drop_duplicates(subset=["tool_name"])
    log.info(f"Fetched {len(df)} unique products from ProductHunt")
    return df

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = os.getenv("PRODUCTHUNT_TOKEN")
    if not token:
        raise EnvironmentError(
            "PRODUCTHUNT_TOKEN not set. "
            "Copy .env.example to .env and add your Developer Token."
        )

    cfg = load_config()
    df = fetch_posts(
        token=token,
        max_posts=cfg["scraping"]["producthunt_max_posts"],
        delay=cfg["scraping"]["producthunt_delay_seconds"],
    )
    out_path = Path("data/raw_producthunt.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Saved to {out_path}")
    print(df.head(10).to_string())
