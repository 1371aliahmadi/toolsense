"""
Label Studio Preparation — Phase 08
======================================
Generates annotation tasks for Label Studio and converts
exported annotations back into evaluation ground truth.

Workflow:
  1. run_phase08.py generates tasks JSON  →  import into Label Studio
  2. You annotate: for each query, mark which tools are relevant
  3. Export from Label Studio as JSON  →  saved to data/annotations_export.json
  4. run_phase08.py --convert reads the export and writes data/ground_truth.json
  5. Phase 06 / 07 runners can then use real ground truth instead of the proxy set

Label Studio setup:
  pip install label-studio
  label-studio start          # opens http://localhost:8080
"""

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ── 100 diverse annotation queries ───────────────────────────────────────────
# Covers the same categories as the 20 proxy queries but with more variety
# and phrasing styles — short, long, vague, specific, with typos sometimes.

ANNOTATION_QUERIES = [
    # Video / Audio
    "generate AI-powered videos for YouTube",
    "text to video generation tool",
    "AI video editor for short clips",
    "create talking avatar from text",
    "convert blog post to video automatically",
    "transcribe audio and video to text",
    "generate podcast from text",
    "voice cloning for content creation",
    "AI dubbing tool for translation",
    "automatic captions and subtitles",
    # Writing / Content
    "write better marketing copy and emails",
    "AI writing assistant for blog posts",
    "long-form content generator for articles",
    "grammar and style checker for documents",
    "paraphrase and rewrite sentences",
    "AI tool for academic writing and citations",
    "generate product descriptions for e-commerce",
    "creative writing and storytelling AI",
    "AI ghostwriter for social media bios",
    "generate press releases automatically",
    # Images / Design
    "generate images from text prompts",
    "AI tool for graphic design and logos",
    "background remover for photos",
    "AI interior design and room decoration",
    "create infographics from data automatically",
    "upscale and enhance low-resolution images",
    "AI fashion and clothing design tool",
    "generate illustrations for children books",
    "3D model generation from text",
    "color palette generator for designers",
    # Code / Developer
    "automate coding and fix bugs",
    "code review and debugging assistant",
    "generate unit tests automatically",
    "AI pair programmer for Python",
    "convert natural language to SQL queries",
    "auto-document code and write comments",
    "API integration and backend automation",
    "DevOps and infrastructure automation tool",
    "security vulnerability scanner for code",
    "AI for database query optimization",
    # Marketing / Sales / SEO
    "SEO optimization for my website",
    "create social media posts automatically",
    "sales outreach and cold email automation",
    "monitor social media mentions",
    "AI for keyword research and content planning",
    "competitor analysis and market research",
    "ad copy generation for Google Ads",
    "A/B testing content for landing pages",
    "influencer marketing campaign tool",
    "generate hashtags for Instagram posts",
    # Customer Support / Chat
    "customer support chatbot for my website",
    "AI helpdesk and ticket routing",
    "FAQ bot builder without coding",
    "live chat with AI fallback",
    "sentiment analysis for customer feedback",
    "voice AI for inbound customer calls",
    "email triage and auto-response",
    "customer onboarding automation",
    "multilingual support chatbot",
    "AI for complaint handling and escalation",
    # Productivity / Business
    "scheduling and calendar management",
    "data analysis and business intelligence",
    "summarize long documents and PDFs",
    "meeting notes and action items from recording",
    "AI executive assistant for inbox management",
    "workflow automation without coding",
    "project management AI assistant",
    "AI for financial forecasting and budgeting",
    "contract review and legal document analysis",
    "knowledge base and internal wiki builder",
    # HR / Recruiting
    "HR recruiting and candidate screening",
    "AI resume screener and ranking",
    "generate job descriptions automatically",
    "employee onboarding automation",
    "performance review assistant",
    "team scheduling and shift planning",
    "AI for employee engagement surveys",
    "learning and development course generator",
    "salary benchmarking tool",
    "AI reference check automation",
    # Language / Translation
    "translate text between languages",
    "AI language learning tutor",
    "real-time speech translation",
    "localize website content for multiple markets",
    "transcribe and translate meetings",
    # Presentations / Docs
    "create presentations and slide decks",
    "AI for creating pitch decks for startups",
    "generate executive summary from report",
    "document comparison and diff tool",
    "convert PDF to editable Word document",
    # Research / Data
    "web scraping and data extraction AI",
    "AI research assistant for literature review",
    "survey analysis and insights generation",
    "AI for market trend analysis",
    "automated reporting from spreadsheets",
    # Misc / Emerging
    "AI for personal finance and budgeting",
    "AI recipe generator and meal planner",
    "generate business plan from idea",
    "AI for real estate property valuation",
    "no-code app builder with AI",
    # Healthcare / Medical
    "AI medical diagnosis assistant",
    "mental health chatbot and therapy support",
    "drug interaction checker with AI",
    "AI for clinical note taking",
    "patient symptom checker tool",
    "AI nutrition and diet planning",
    "fitness coaching and workout planner AI",
    "AI for medical image analysis",
    "telemedicine and remote patient monitoring",
    "AI for hospital operations and scheduling",
    # Education / E-learning
    "AI tutor for math and science",
    "generate quiz and exam questions automatically",
    "AI for language pronunciation practice",
    "personalized learning path generator",
    "AI essay grader for teachers",
    "create online course content with AI",
    "AI flashcard generator from notes",
    "coding tutor for beginners",
    "AI for special education and accessibility",
    "generate lesson plans automatically",
    # Data Science / ML
    "AI for data cleaning and preprocessing",
    "automate machine learning model training",
    "AI for time series forecasting",
    "anomaly detection in datasets",
    "AI feature engineering assistant",
    "explain machine learning model predictions",
    "AI for A/B test analysis",
    "automated data visualization from CSV",
    "AI for ETL pipeline automation",
    "natural language interface for databases",
    # E-commerce / Retail
    "AI product recommendation engine",
    "dynamic pricing tool for online store",
    "AI for inventory management",
    "generate product titles and SEO descriptions",
    "AI chatbot for Shopify store",
    "visual search for fashion e-commerce",
    "AI for returns and refund automation",
    "customer lifetime value prediction",
    "AI for abandoned cart recovery",
    "review analysis and response automation",
    # Audio / Music Production
    "AI music composition from scratch",
    "stem separation and audio isolation",
    "AI mastering and mixing tool",
    "generate sound effects from text",
    "AI DJ and playlist curator",
    "voice enhancement and noise removal",
    "AI for podcast editing and cleanup",
    "MIDI generation from humming",
    "AI lyric writing assistant",
    "audio transcription with speaker diarization",
    # Legal / Compliance
    "AI contract drafting and review",
    "GDPR compliance checker",
    "AI for intellectual property research",
    "legal research and case law finder",
    "AI for terms and conditions generation",
    "compliance monitoring automation",
    "AI paralegal assistant",
    "trademark and patent search AI",
    "AI for regulatory document analysis",
    "due diligence automation tool",
    # Finance / Fintech
    "AI stock market analysis and prediction",
    "automated accounting and bookkeeping",
    "AI for fraud detection",
    "personal investment portfolio optimizer",
    "AI tax preparation assistant",
    "expense categorization from receipts",
    "AI for loan underwriting",
    "cryptocurrency trading bot",
    "financial report generation from data",
    "AI for insurance claims processing",
    # Vague / Short / Edge-case queries
    "make videos with AI",
    "write emails faster",
    "fix my code",
    "talk to AI",
    "edit photos",
    "summarize this",
    "translate document",
    "find leads",
    "automate tasks",
    "analyze data",
    # Typo / Misspelling queries
    "trasncribe my meating",
    "genrate images from text",
    "custmer suport chatbot",
    "seo keword reserch",
    "video editng tool",
    # Very specific / niche
    "AI for Dungeons and Dragons dungeon master",
    "generate tattoo designs with AI",
    "AI for wine pairing recommendations",
    "dream interpretation AI",
    "AI for astrology and horoscope generation",
    # Additional coverage
    "AI for sports performance analytics",
    "generate children story with illustrations",
    "AI interior architect for home renovation",
    "AI for supply chain optimization",
    "turn handwritten notes into digital text",
    "AI for grant writing and fundraising",
    "generate memes and viral content with AI",
    "AI for event planning and management",
    "detect plagiarism in documents",
    "AI for news summarization and briefing",
    "AI avatar for virtual meetings",
    "generate social media calendar automatically",
    "AI for scientific research paper writing",
    "chatbot for mental wellness and journaling",
    "AI for customer churn prediction",
    "generate wireframes and UI mockups with AI",
    "AI for cybersecurity threat detection",
    "voice assistant for smart home devices",
    "AI for travel itinerary planning",
    "generate subtitles and closed captions",
]

assert len(ANNOTATION_QUERIES) == 210, f"Expected 210 queries, got {len(ANNOTATION_QUERIES)}"


# ── Generate Label Studio tasks ───────────────────────────────────────────────

def generate_tasks(retrieve_fn, qdrant_path: str, weights: dict, top_k: int = 20) -> list[dict]:
    """
    For each annotation query, retrieve top_k tools and package as a
    Label Studio task with the query and candidate tools shown to the annotator.
    """
    tasks = []

    for i, query in enumerate(ANNOTATION_QUERIES):
        try:
            results = retrieve_fn(
                query=query,
                qdrant_path=qdrant_path,
                top_k=top_k,
                return_k=top_k,   # show all top_k to annotator
                **weights,
            )
        except Exception as e:
            log.warning(f"Retrieval failed for '{query}': {e}")
            results = []

        # Each choice is one candidate tool
        choices = [
            {
                "value": r["tool_name"],
                "label": f"{r['tool_name']} | {r['category']} | Score: {r['final_score']:.3f}",
                "description": r["description"][:150],
            }
            for r in results
        ]

        # Flat list of tool names — used by Label Studio dynValue choices
        tool_names = [r["tool_name"] for r in results]

        # Rich text shown to annotator (query + all candidates with descriptions)
        candidates_text = "\n".join(
            f"{idx+1}. {r['tool_name']} [{r['category']}] — {r['description'][:100]}"
            for idx, r in enumerate(results)
        )

        task = {
            "id": i + 1,
            "data": {
                "query": query,
                "instruction": (
                    "Tick ALL tools below that a user searching for this query "
                    "would find genuinely useful."
                ),
                # dynValue list — Label Studio renders these as checkboxes
                "choices": tool_names,
                # Human-readable reference shown above the checkboxes
                "candidates_text": candidates_text,
                # Store full results for later reference
                "_results_json": json.dumps(results),
            },
        }
        tasks.append(task)
        if (i + 1) % 10 == 0:
            print(f"  Prepared {i + 1}/{len(ANNOTATION_QUERIES)} tasks...")

    return tasks


def save_tasks(tasks: list[dict], out_path: Path) -> None:
    """Save tasks as Label Studio import JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(tasks)} tasks to {out_path}")


# ── Label Studio labeling config (XML) ───────────────────────────────────────

LABEL_CONFIG = """<View>
  <Header value="Query: $query" style="font-size:1.4em; font-weight:bold; color:#2c3e50"/>
  <Text name="instruction" value="$instruction" style="color:#7f8c8d; margin-bottom:12px"/>
  <Choices name="relevant_tools" toName="query" choice="multiple" showInline="false">
    <Choice value="{{value}}" hint="{{description}}"/>
  </Choices>
</View>"""


# ── Convert Label Studio export → ground truth ────────────────────────────────

def convert_export(export_path: Path) -> list[dict]:
    """
    Read a Label Studio JSON export and convert to ground-truth format:
      [{"query": str, "relevant_tools": [str, ...]}, ...]

    Annotators type comma-separated tool numbers in a TextArea (e.g. "1, 5, 9").
    We map those numbers back to tool names using the task's `choices` list.
    """
    with open(export_path, encoding="utf-8") as f:
        export = json.load(f)

    ground_truth = []
    skipped = 0

    for item in export:
        query      = item["data"]["query"]
        tool_names = item["data"].get("choices", [])   # ordered list of 20 tool names
        relevant   = []

        annotations = item.get("annotations", [])
        if annotations:
            results = annotations[0].get("result", [])
            for r in results:
                # TextArea annotation: value.text is a list of strings
                texts = r.get("value", {}).get("text", [])
                for text in texts:
                    # Parse "1, 5, 9" or "1,5,9" or "1 5 9"
                    import re
                    nums = re.findall(r"\d+", text)
                    for n in nums:
                        idx = int(n) - 1   # 1-indexed → 0-indexed
                        if 0 <= idx < len(tool_names):
                            relevant.append(tool_names[idx])
                        else:
                            log.warning(f"Number {n} out of range for query '{query}'")

        if not relevant:
            skipped += 1

        ground_truth.append({
            "query":          query,
            "relevant_tools": list(dict.fromkeys(relevant)),  # deduplicate, keep order
        })

    if skipped:
        log.info(f"{skipped} tasks had no annotation (skipped or unannotated)")

    return ground_truth


def save_ground_truth(ground_truth: list[dict], out_path: Path) -> None:
    """Save converted ground truth to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    n_labeled = sum(1 for g in ground_truth if g["relevant_tools"])
    avg_rel   = (
        sum(len(g["relevant_tools"]) for g in ground_truth) / max(n_labeled, 1)
    )
    print(f"Ground truth saved: {len(ground_truth)} queries")
    print(f"  Queries with at least 1 label: {n_labeled}")
    print(f"  Avg relevant tools per query:  {avg_rel:.1f}")


# ── Load ground truth for evaluation ─────────────────────────────────────────

def load_ground_truth(gt_path: Path) -> list[tuple[str, set[str]]]:
    """
    Load saved ground truth as a list of (query, relevant_tool_names_set) tuples
    — same format as PROXY_TEST_SET, so it can be dropped in directly.
    """
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        (item["query"], set(item["relevant_tools"]))
        for item in data
        if item["relevant_tools"]   # skip unannotated
    ]
