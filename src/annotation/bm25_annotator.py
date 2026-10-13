"""
BM25 Auto-Annotator — Phase 08 (no LLM / no API key needed)
=============================================================
Generates ground-truth annotations using BM25 keyword search over the full
clean_tools.csv catalogue.

Why BM25 instead of an LLM?
  - Completely independent of SBERT (no circular evaluation)
  - No API key or internet required
  - Deterministic and reproducible
  - BM25 and SBERT measure orthogonal signals, so BM25 ground truth is valid

Strategy per query:
  1. Tokenise query + tool texts (name · category · description)
  2. Score all 2,121 tools with BM25
  3. Mark tools as relevant if:
       BM25_score > 0 AND BM25_score >= RELEVANCE_RATIO * top_score
       (keeps tools with at least RELEVANCE_RATIO of the best match)
  4. Cap at MAX_RELEVANT per query (prevents annotation noise)
  5. Require at least MIN_RELEVANT (queries with 0 are kept but flagged)

Output:
  data/label_studio_tasks.json   — 210 tasks with BM25 top-20 candidates
  data/annotations_export.json   — same format as LLM annotator / Label Studio export

Usage:
    python src/annotation/bm25_annotator.py
    python run_phase08.py --convert
"""

import json
import re
import logging
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [BM25] %(message)s")
log = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parents[2]
CLEAN_CSV     = ROOT / "data" / "clean_tools.csv"
TASKS_PATH    = ROOT / "data" / "label_studio_tasks.json"
OUTPUT_PATH   = ROOT / "data" / "annotations_export.json"

TOP_K         = 20      # candidates shown per query
RELEVANCE_RATIO = 0.20  # mark tool as relevant if score >= 20% of best score
MAX_RELEVANT  = 10      # cap relevant per query (avoid trivial all-positive)
MIN_SCORE     = 0.01    # ignore near-zero scores (no real keyword overlap)

STOPWORDS = {
    "a", "an", "the", "for", "and", "or", "to", "of", "in", "my",
    "from", "with", "that", "this", "is", "are", "as", "i", "me",
    "can", "do", "how", "what", "which", "using", "use", "tool",
    "ai", "ml", "platform", "app", "software", "solution",
}


def tokenise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return tokens


def build_corpus(df: pd.DataFrame) -> list[list[str]]:
    """Build tokenised corpus: name (×3 weight) + category (×2) + description."""
    corpus = []
    for _, row in df.iterrows():
        name   = str(row.get("tool_name", "") or "")
        cat    = str(row.get("category", "") or "")
        desc   = str(row.get("description", "") or "")
        tokens = (
            tokenise(name) * 3    # boost tool name match
            + tokenise(cat) * 2   # boost category match
            + tokenise(desc)
        )
        corpus.append(tokens if tokens else ["__empty__"])
    return corpus


def annotate(
    df: pd.DataFrame,
    bm25: BM25Okapi,
    queries: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    For each query:
      - Retrieve top-K candidates by BM25
      - Mark relevant if score >= RELEVANCE_RATIO * max_score and score > MIN_SCORE

    Returns:
      tasks  : list of Label-Studio-format task dicts
      export : list of annotation export dicts (same format as llm_annotator)
    """
    tool_names = df["tool_name"].tolist()
    tasks      = []
    export     = []

    for i, query in enumerate(queries):
        q_tokens = tokenise(query)
        if not q_tokens:
            q_tokens = query.lower().split()

        scores = bm25.get_scores(q_tokens)

        # Get top-K indices by score
        top_indices = sorted(range(len(scores)), key=lambda x: -scores[x])[:TOP_K]

        # Build candidate list
        candidates = []
        for idx in top_indices:
            row = df.iloc[idx]
            candidates.append({
                "tool_name":    row["tool_name"],
                "category":     str(row.get("category", "") or ""),
                "description":  str(row.get("description", "") or "")[:150],
                "bm25_score":   float(scores[idx]),
                "final_score":  float(scores[idx]),   # field expected by label_studio_prep
                "source":       str(row.get("source", "") or ""),
            })

        if not candidates:
            continue

        max_score = candidates[0]["bm25_score"]

        # Determine relevant tools
        relevant_names = []
        relevant_indices = []  # 1-based
        for rank, c in enumerate(candidates, start=1):
            is_relevant = (
                c["bm25_score"] >= max_score * RELEVANCE_RATIO
                and c["bm25_score"] > MIN_SCORE
                and len(relevant_names) < MAX_RELEVANT
            )
            if is_relevant:
                relevant_names.append(c["tool_name"])
                relevant_indices.append(rank)

        candidate_names = [c["tool_name"] for c in candidates]
        candidates_text = "\n".join(
            f"{rank}. {c['tool_name']} [{c['category']}] — {c['description'][:100]}"
            for rank, c in enumerate(candidates, start=1)
        )

        # ── Label Studio task ─────────────────────────────────────────────────
        task = {
            "id": i + 1,
            "data": {
                "query": query,
                "instruction": (
                    "Tick ALL tools below that a user searching for this query "
                    "would find genuinely useful."
                ),
                "choices":         candidate_names,
                "candidates_text": candidates_text,
                "_results_json":   json.dumps(candidates),
            },
        }
        tasks.append(task)

        # ── Export entry (same format as llm_annotator) ───────────────────────
        export_entry = {
            "id":   i + 1,
            "data": task["data"],
            "annotations": [
                {
                    "result": [
                        {
                            "type":      "textarea",
                            "value":     {
                                "text": [
                                    ", ".join(str(idx) for idx in relevant_indices)
                                    if relevant_indices else ""
                                ]
                            },
                            "from_name": "relevant_numbers",
                            "to_name":   "ref",
                        }
                    ],
                    "ground_truth": False,
                }
            ],
            "_llm_response":  ", ".join(str(idx) for idx in relevant_indices),
            "_relevant_tools": relevant_names,
        }
        export.append(export_entry)

        if (i + 1) % 20 == 0 or (i + 1) == len(queries):
            log.info(f"  [{i+1}/{len(queries)}] '{query[:50]}' → {len(relevant_names)} relevant")

    return tasks, export


def main():
    # Load catalogue
    log.info(f"Loading clean tools from {CLEAN_CSV}")
    df = pd.read_csv(CLEAN_CSV)
    log.info(f"  → {len(df)} tools loaded")

    # Build BM25 index
    log.info("Building BM25 index …")
    corpus = build_corpus(df)
    bm25   = BM25Okapi(corpus)
    log.info("  → Index ready")

    # Load queries — parse source directly to avoid stale .pyc cache issues
    prep_path = ROOT / "src" / "annotation" / "label_studio_prep.py"
    queries = []
    in_list = False
    for line in open(prep_path, encoding="utf-8", errors="replace"):
        s = line.strip()
        if "ANNOTATION_QUERIES = [" in line:
            in_list = True
            continue
        if in_list:
            if s == "]":
                break
            if s.startswith('"') and (s.endswith('",') or s.endswith('"')):
                queries.append(s.strip('",').strip())
    log.info(f"Loaded {len(queries)} queries from label_studio_prep.py")
    log.info(f"Annotating {len(queries)} queries …")

    # Run annotation
    tasks, export = annotate(df, bm25, queries)

    # Save tasks.json
    TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TASKS_PATH, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    log.info(f"Tasks saved: {TASKS_PATH}  ({len(tasks)} tasks)")

    # Save annotations_export.json
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    log.info(f"Annotations saved: {OUTPUT_PATH}  ({len(export)} annotations)")

    # Summary
    n_with_labels = sum(1 for e in export if e["_relevant_tools"])
    total_rel     = sum(len(e["_relevant_tools"]) for e in export)
    avg_rel       = total_rel / max(n_with_labels, 1)

    print("\n" + "=" * 55)
    print("BM25 ANNOTATION COMPLETE")
    print("=" * 55)
    print(f"  Queries annotated:               {len(export)}")
    print(f"  Queries with ≥1 relevant tool:   {n_with_labels}/{len(export)}")
    print(f"  Total relevant labels:           {total_rel}")
    print(f"  Avg relevant tools per query:    {avg_rel:.1f}")
    print(f"  Tasks  → {TASKS_PATH}")
    print(f"  Export → {OUTPUT_PATH}")
    print("=" * 55)
    print("\nNext step:  python run_phase08.py --convert")


if __name__ == "__main__":
    main()
