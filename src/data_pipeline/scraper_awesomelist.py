"""
Awesome AI Tools Scraper
=========================
Parses the GitHub awesome list:
  https://github.com/mahseema/awesome-ai-tools

Format in README.md:
  ## Category Name
  - [Tool Name](https://url.com) - One-line description of the tool.

No API key needed — fetches the raw markdown from GitHub.

Usage:
    python -m src.data_pipeline.scraper_awesomelist
"""

import re
import logging
import requests
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AwesomeList] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

README_URL = "https://raw.githubusercontent.com/mahseema/awesome-ai-tools/main/README.md"
OUT_PATH   = Path(__file__).resolve().parents[2] / "data" / "raw_awesomelist.csv"

# Sections to skip (meta / not actual tools)
SKIP_SECTIONS = {
    "contents", "editor's choice", "learning resources",
    "contributing", "license", "other resources",
}

# Map awesome-list section headings → ToolSense category names
SECTION_TO_CATEGORY = {
    "text":           "Writing, Text Generation",
    "ai text":        "Writing, Text Generation",
    "code":           "Code, Developer Tools",
    "code with ai":   "Code, Developer Tools",
    "image":          "Image Generation, Design",
    "generative ai images": "Image Generation, Design",
    "video":          "Video Generation",
    "generative ai video":  "Video Generation",
    "audio":          "Audio, Music",
    "generative ai audio":  "Audio, Music",
    "voice cloning":  "Audio, Voice Cloning",
    "ai voice cloning": "Audio, Voice Cloning",
    "music generation": "Audio, Music Generation",
    "ai music generators": "Audio, Music Generation",
    "marketing":      "Marketing, SEO",
    "marketing ai tools": "Marketing, SEO",
    "phone calls":    "Customer Support, Voice AI",
    "ai phone call agents": "Customer Support, Voice AI",
    "other":          "Productivity, AI Tools",
    "other ai tools": "Productivity, AI Tools",
}


@dataclass
class AwesomeTool:
    tool_name:    str
    description:  str
    category:     str
    pricing_tier: str = "Unknown"
    upvotes:      int = 0
    url:          str = ""
    source:       str = "awesomelist"


def fetch_readme() -> str:
    log.info(f"Fetching README from {README_URL}")
    resp = requests.get(README_URL, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_readme(md: str) -> list[AwesomeTool]:
    tools = []
    current_category = "AI Tools"
    seen_urls = set()

    # Regex: matches markdown list items like:
    #   - [Name](url) - Description
    #   - [Name](url): Description
    #   * [Name](url) - Description
    item_re = re.compile(
        r'^[\-\*]\s+\[([^\]]+)\]\((https?://[^)]+)\)\s*[-–:]\s*(.+)$',
        re.MULTILINE,
    )
    # Regex: matches section headers ## Heading or ### Heading
    header_re = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)

    # Walk through the markdown line by line tracking current section
    lines = md.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for section header
        h_match = re.match(r'^#{1,3}\s+(.+)$', line)
        if h_match:
            raw_heading = h_match.group(1).strip()
            # Strip emoji and badges from heading
            clean = re.sub(r'[^\w\s]', ' ', raw_heading).strip().lower()
            clean = re.sub(r'\s+', ' ', clean).strip()
            if clean in SKIP_SECTIONS:
                current_category = None
            else:
                current_category = SECTION_TO_CATEGORY.get(
                    clean,
                    raw_heading.strip().title()
                )

        # Check for tool item
        if current_category and line.startswith(('-', '*')):
            m = item_re.match(line)
            if m:
                name = m.group(1).strip()
                url  = m.group(2).strip()
                desc = m.group(3).strip()

                # Skip if it's a directory/list link (not a tool)
                if url in seen_urls:
                    i += 1; continue
                if any(kw in name.lower() for kw in ['list of', 'awesome', 'directory', 'newsletter']):
                    i += 1; continue
                if len(desc) < 10:
                    i += 1; continue

                # Clean up description
                desc = desc.rstrip('.')
                if len(desc) > 500:
                    desc = desc[:500]

                seen_urls.add(url)
                tools.append(AwesomeTool(
                    tool_name=name,
                    description=desc,
                    category=current_category,
                    url=url,
                ))

        i += 1

    return tools


def run():
    md = fetch_readme()
    tools = parse_readme(md)

    log.info(f"Parsed {len(tools)} tools from awesome-ai-tools")

    # Category breakdown
    from collections import Counter
    cats = Counter(t.category for t in tools)
    for cat, count in cats.most_common():
        log.info(f"  {cat}: {count}")

    df = pd.DataFrame([asdict(t) for t in tools])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    log.info(f"Saved → {OUT_PATH}")
    return df


if __name__ == "__main__":
    df = run()
    print(f"\nTotal tools scraped: {len(df)}")
    print(df[["tool_name", "category", "url"]].head(10).to_string())
