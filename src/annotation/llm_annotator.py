"""
LLM-as-Annotator — Phase 08
==============================
Uses Ollama (local), GPT-4o, or Gemini to automatically annotate
relevance labels for all 100 annotation queries.

This is a legitimate research technique (LLM-as-annotator / synthetic annotation).
LLMs agree with human annotators ~80-90% on relevance tasks.

Usage:
    # Default — Ollama (local, no API key needed):
    python src/annotation/llm_annotator.py
    python src/annotation/llm_annotator.py --provider ollama --model llama3.2:3b

    # With OpenAI (GPT-4o):
    set OPENAI_API_KEY=sk-...
    python src/annotation/llm_annotator.py --provider openai

    # With Google Gemini:
    set GEMINI_API_KEY=...
    python src/annotation/llm_annotator.py --provider gemini --model gemini-2.0-flash

Output: data/annotations_export.json  (same format as manual Label Studio export)
Then run: python run_phase08.py --convert
"""

import json
import time
import argparse
import os
import re
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TASKS_PATH  = Path(__file__).resolve().parents[2] / "data" / "label_studio_tasks.json"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "annotations_export.json"

SYSTEM_PROMPT = """You are an expert at evaluating whether AI tools are relevant to a user's search query.

You will be given:
1. A search query (what the user is looking for)
2. A numbered list of AI tools with their categories and descriptions

Your task: identify which tools are genuinely relevant to the query.
A tool is relevant if a user searching for that query would find it useful.
Be strict — only mark tools as relevant if they clearly match the intent of the query.

Respond ONLY with a comma-separated list of numbers (e.g. "3, 7, 12, 15").
If none are relevant, respond with "none".
Do not explain your reasoning. Just the numbers."""


def build_prompt(query: str, candidates_text: str) -> str:
    return f"""Query: "{query}"

Candidate tools:
{candidates_text}

Which tool numbers are relevant to this query? Reply with only the numbers."""


def call_openai(client, query: str, candidates_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # cheap and fast; use gpt-4o for higher quality
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, candidates_text)},
        ],
        temperature=0.0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def call_gemini(model, query: str, candidates_text: str) -> str:
    prompt = SYSTEM_PROMPT + "\n\n" + build_prompt(query, candidates_text)
    response = model.generate_content(prompt)
    return response.text.strip()


def parse_numbers(response: str, n_candidates: int) -> list[int]:
    """Parse LLM response into list of 1-based indices."""
    if "none" in response.lower():
        return []
    nums = re.findall(r"\d+", response)
    valid = []
    for n in nums:
        idx = int(n)
        if 1 <= idx <= n_candidates:
            valid.append(idx)
        else:
            log.warning(f"LLM returned out-of-range number: {n} (max {n_candidates})")
    return valid


def annotate_with_llm(
    tasks: list[dict],
    call_fn,
    output_path: Path = OUTPUT_PATH,
    delay: float = 5.0,
) -> list[dict]:
    """
    Run LLM annotation over all tasks with incremental saves and resume support.
    Already-annotated tasks (by id) in output_path are skipped.
    """
    # Load partial progress if exists
    done_ids = set()
    export = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            export = json.load(f)
        done_ids = {e["id"] for e in export}
        log.info(f"Resuming: {len(done_ids)} tasks already done, skipping them.")

    n = len(tasks)
    remaining = [t for t in tasks if t["id"] not in done_ids]
    log.info(f"Tasks remaining: {len(remaining)}/{n}")

    for i, task in enumerate(remaining):
        query          = task["data"]["query"]
        candidates_text= task["data"]["candidates_text"]
        tool_names     = task["data"]["choices"]

        indices  = []
        relevant = []
        response = ""

        try:
            response = call_fn(query, candidates_text)
            indices  = parse_numbers(response, len(tool_names))
            relevant = [tool_names[idx - 1] for idx in indices]
            log.info(f"[{len(done_ids)+i+1}/{n}] '{query[:50]}' -> {len(relevant)} relevant")

        except Exception as e:
            log.error(f"[{len(done_ids)+i+1}/{n}] Failed for '{query}': {e}")
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                log.info("Rate limited — waiting 60s then retrying...")
                time.sleep(60)
                try:
                    response = call_fn(query, candidates_text)
                    indices  = parse_numbers(response, len(tool_names))
                    relevant = [tool_names[idx - 1] for idx in indices]
                    log.info(f"  Retry succeeded: {len(relevant)} relevant")
                except Exception as e2:
                    log.error(f"  Retry also failed: {e2}")

        entry = {
            "id":   task["id"],
            "data": task["data"],
            "annotations": [
                {
                    "result": [
                        {
                            "type":      "textarea",
                            "value":     {
                                "text": [
                                    ", ".join(str(idx) for idx in indices)
                                    if indices else ""
                                ]
                            },
                            "from_name": "relevant_numbers",
                            "to_name":   "ref",
                        }
                    ],
                    "ground_truth": False,
                }
            ],
            "_llm_response":  response,
            "_relevant_tools": relevant,
        }
        export.append(entry)

        # Save after every single query (incremental)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)

        if (len(done_ids) + i + 1) % 10 == 0:
            print(f"  Progress: {len(done_ids)+i+1}/{n} queries annotated...")

        time.sleep(delay)

    return export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "gemini", "ollama"], default="ollama")
    parser.add_argument("--model",    default=None,
                        help="Override model (e.g. gpt-4o, gemini-2.0-flash, llama3.2:3b)")
    parser.add_argument("--delay",    type=float, default=5.0,
                        help="Seconds between API calls (default 5.0)")
    args = parser.parse_args()

    # Load tasks
    if not TASKS_PATH.exists():
        print(f"ERROR: Tasks file not found: {TASKS_PATH}")
        print("Run `python run_phase08.py` first to generate tasks.")
        return

    with open(TASKS_PATH, encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks from {TASKS_PATH}")

    # Set up LLM client
    if args.provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("Install openai: pip install openai")
            return

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("Set OPENAI_API_KEY environment variable first.")
            return

        client = OpenAI(api_key=api_key)
        model_name = args.model or "gpt-4o-mini"
        print(f"Using OpenAI model: {model_name}")

        def call_fn(query, candidates_text):
            return call_openai(client, query, candidates_text)

    elif args.provider == "gemini":
        try:
            from google import genai as google_genai
        except ImportError:
            print("Install google-genai: pip install google-genai")
            return

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("Set GEMINI_API_KEY environment variable first.")
            return

        client_genai = google_genai.Client(api_key=api_key)
        model_name = args.model or "gemini-2.0-flash"
        print(f"Using Gemini model: {model_name}")

        def call_fn(query, candidates_text):
            prompt = SYSTEM_PROMPT + "\n\n" + build_prompt(query, candidates_text)
            response = client_genai.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text.strip()

    else:  # ollama
        import urllib.request
        import json as _json

        ollama_url  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name  = args.model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        print(f"Using Ollama model: {model_name} at {ollama_url}")

        def call_fn(query, candidates_text):
            prompt = SYSTEM_PROMPT + "\n\n" + build_prompt(query, candidates_text)
            payload = _json.dumps({
                "model":  model_name,
                "prompt": prompt,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = _json.loads(resp.read())
            return result["response"].strip()

    # Run annotation
    print(f"\nAnnotating {len(tasks)} queries with {args.provider.upper()}...")
    print("(This takes ~1-2 minutes with rate limiting)\n")

    export = annotate_with_llm(tasks, call_fn, delay=args.delay)

    # Save export
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    # Summary
    n_labeled  = sum(1 for e in export if e["_relevant_tools"])
    total_rels = sum(len(e["_relevant_tools"]) for e in export)
    avg_rel    = total_rels / max(n_labeled, 1)

    print(f"\nDone! Annotated {len(export)} queries.")
    print(f"  Queries with >= 1 relevant tool: {n_labeled}/{len(export)}")
    print(f"  Avg relevant tools per query:    {avg_rel:.1f}")
    print(f"  Output saved to: {OUTPUT_PATH}")
    print(f"\nNext step: python run_phase08.py --convert")


if __name__ == "__main__":
    main()
