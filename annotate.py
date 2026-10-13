"""
Interactive Annotator — ToolSense
===================================
Lets you annotate queries one by one in the terminal.
Press ENTER to skip, type numbers to mark relevant, q to quit.

Usage:
    python annotate.py                   # annotate from the start
    python annotate.py --from 50         # start from query #50
    python annotate.py --queries 20      # do only 20 queries, then stop

Output: data/annotations_export.json  (saves after each query)
Then:   python run_phase08.py --convert

Controls per query:
    1 3 7       → mark tools 1, 3, and 7 as relevant (space-separated)
    1-5         → mark tools 1 through 5
    all         → mark all 20 as relevant
    none / n    → none are relevant
    s / ENTER   → skip this query (no annotation saved)
    q           → quit and save progress
"""

import json
import sys
import re
import argparse
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
TASKS_PATH  = ROOT / "data" / "label_studio_tasks.json"
OUT_PATH    = ROOT / "data" / "annotations_export.json"

COLORS = {
    "header":  "\033[1;36m",   # bold cyan
    "query":   "\033[1;33m",   # bold yellow
    "num":     "\033[1;32m",   # bold green
    "cat":     "\033[0;35m",   # magenta
    "desc":    "\033[0;37m",   # light grey
    "prompt":  "\033[1;34m",   # bold blue
    "saved":   "\033[0;32m",   # green
    "skip":    "\033[0;33m",   # yellow
    "reset":   "\033[0m",
}

def c(key, text):
    return f"{COLORS[key]}{text}{COLORS['reset']}"


def parse_numbers(response: str, n: int) -> list[int]:
    """Parse '1 3 7' or '1-5' or 'all' into 1-based indices."""
    response = response.strip().lower()
    if not response or response in ("s", "skip", ""):
        return None          # skip
    if response in ("n", "none", "0"):
        return []
    if response == "all":
        return list(range(1, n + 1))

    # expand ranges like "1-5"
    expanded = re.sub(r"(\d+)-(\d+)", lambda m: " ".join(
        str(x) for x in range(int(m.group(1)), int(m.group(2)) + 1)
    ), response)

    nums = []
    for tok in re.findall(r"\d+", expanded):
        idx = int(tok)
        if 1 <= idx <= n:
            nums.append(idx)
        else:
            print(f"  ⚠  Number {idx} out of range (1-{n}), ignoring.")
    return nums


def load_progress() -> tuple[list[dict], set[int]]:
    """Load existing export if present (resume support)."""
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            export = json.load(f)
        done_ids = {e["id"] for e in export}
        return export, done_ids
    return [], set()


def save(export: list[dict]):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)


def annotate_task(task: dict, idx: int, total: int) -> dict | None:
    """
    Show one task and get user input.
    Returns annotation dict or None if skipped.
    """
    query      = task["data"]["query"]
    tool_names = task["data"]["choices"]
    candidates_text = task["data"].get("candidates_text", "")
    n = len(tool_names)

    print()
    print("─" * 70)
    print(c("num", f"  [{idx}/{total}]  ") + c("query", f'"{query}"'))
    print("─" * 70)

    # Show candidates
    lines = candidates_text.strip().split("\n") if candidates_text else []
    for line in lines:
        # Colour the number prefix
        m = re.match(r"^(\d+)\.\s+(.*)\[([^\]]*)\]\s*—\s*(.*)", line)
        if m:
            num, name, cat, desc = m.groups()
            print(
                c("num",  f"  {num:>2}. ") +
                f"{name} " +
                c("cat",  f"[{cat}]") +
                c("desc", f" — {desc[:90]}")
            )
        else:
            print(f"  {line}")

    print()
    print(c("prompt", "  Relevant? ") +
          "(numbers, ranges '1-5', 'all', 'none'/'n', ENTER=skip, 'q'=quit)")
    try:
        response = input(c("prompt", "  > ")).strip()
    except (EOFError, KeyboardInterrupt):
        return "quit"

    if response.lower() == "q":
        return "quit"

    indices = parse_numbers(response, n)
    if indices is None:
        print(c("skip", "  ↷ Skipped"))
        return None

    relevant_names = [tool_names[i - 1] for i in indices]
    print(c("saved", f"  ✓ Saved {len(relevant_names)} relevant tool(s)"))

    return {
        "id":   task["id"],
        "data": task["data"],
        "annotations": [
            {
                "result": [
                    {
                        "type":      "textarea",
                        "value":     {
                            "text": [
                                ", ".join(str(i) for i in indices) if indices else ""
                            ]
                        },
                        "from_name": "relevant_numbers",
                        "to_name":   "ref",
                    }
                ],
                "ground_truth": False,
            }
        ],
        "_llm_response":   ", ".join(str(i) for i in indices),
        "_relevant_tools": relevant_names,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from",     dest="start_from", type=int, default=1,
                        help="Start from query number N (1-based)")
    parser.add_argument("--queries",  type=int, default=None,
                        help="Max queries to annotate in this session")
    args = parser.parse_args()

    if not TASKS_PATH.exists():
        print(f"ERROR: {TASKS_PATH} not found. Run 'python run_phase08.py' first.")
        sys.exit(1)

    with open(TASKS_PATH, encoding="utf-8") as f:
        tasks = json.load(f)

    export, done_ids = load_progress()
    already = len(done_ids)

    print()
    print(c("header", "═" * 70))
    print(c("header", "  ToolSense — Interactive Annotator"))
    print(c("header", "═" * 70))
    print(f"  Total queries:     {len(tasks)}")
    print(f"  Already annotated: {already}")
    print(f"  Remaining:         {len(tasks) - already}")
    print()
    print("  Controls: numbers (e.g. '1 3 7'), ranges ('1-5'),")
    print("            'all', 'none'/'n', ENTER=skip, 'q'=quit")
    print(c("header", "═" * 70))

    count    = 0
    max_q    = args.queries
    start_id = args.start_from   # 1-based

    for task in tasks:
        if task["id"] < start_id:
            continue
        if task["id"] in done_ids:
            print(c("skip", f"  [{task['id']}/{len(tasks)}] already annotated — skipping"))
            continue
        if max_q is not None and count >= max_q:
            print(f"\nDone {max_q} queries for this session.")
            break

        result = annotate_task(task, task["id"], len(tasks))

        if result == "quit":
            print("\n  Saved progress. Resume with: python annotate.py")
            break

        if result is not None:
            export.append(result)
            done_ids.add(task["id"])
            save(export)

        count += 1

    # Final summary
    n_labeled  = sum(1 for e in export if e["_relevant_tools"])
    total_rel  = sum(len(e["_relevant_tools"]) for e in export)

    print()
    print(c("header", "═" * 70))
    print(c("header", "  Session complete"))
    print(c("header", "═" * 70))
    print(f"  Total annotated so far: {len(done_ids)}/{len(tasks)}")
    print(f"  Queries with ≥1 label:  {n_labeled}")
    print(f"  Total relevant labels:  {total_rel}")
    if n_labeled:
        print(f"  Avg relevant per query: {total_rel/n_labeled:.1f}")
    print()
    print(c("saved", "  Next step: python run_phase08.py --convert"))
    print(c("header", "═" * 70))


if __name__ == "__main__":
    main()
