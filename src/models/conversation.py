"""
Conversational Recommender — Phase 04
=======================================
Wraps the ranker + RAG pipeline with a simple chat history
so users can refine their search across multiple turns.

Example session:
  Turn 1: "I need a tool to generate AI videos"
           → returns 5 tools

  Turn 2: "only free ones"
           → filters pricing_tier = "Free", re-ranks

  Turn 3: "which is most popular?"
           → re-sorts by upvotes
"""

import logging
from pathlib import Path

from src.models.ranker import retrieve
from src.models.rag import add_explanations

log = logging.getLogger(__name__)

DEFAULT_QDRANT_PATH = str(
    Path(__file__).resolve().parents[2] / "data" / "qdrant"
)


class ToolSenseConversation:
    """
    Stateful multi-turn recommender session.

    Usage:
        session = ToolSenseConversation()
        results = session.ask("I need a tool to edit videos automatically")
        results = session.ask("only show free ones")
        results = session.ask("which has the most upvotes?")
    """

    def __init__(
        self,
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        w_semantic: float = 0.60,
        w_category: float = 0.25,
        w_popularity: float = 0.15,
        top_k: int = 20,
        return_k: int = 5,
        use_rag: bool = True,
    ):
        self.qdrant_path  = qdrant_path
        self.w_semantic   = w_semantic
        self.w_category   = w_category
        self.w_popularity = w_popularity
        self.top_k        = top_k
        self.return_k     = return_k
        self.use_rag      = use_rag

        # Simple chat history: list of {"role": "user"/"assistant", "content": str}
        self.history: list[dict] = []

        self._last_query   = ""
        self._last_results = []
        self._active_filters: dict = {}

    # ── Parse refinement commands ─────────────────────────────────────────────

    def _parse_filters(self, user_input: str) -> dict:
        text = user_input.lower()
        filters = {}
        if any(w in text for w in ["free", "no cost", "without paying"]):
            filters["pricing_tier"] = "Free"
        elif any(w in text for w in ["paid", "premium", "pro"]):
            filters["pricing_tier"] = "Paid"
        return filters

    def _is_refinement(self, user_input: str) -> bool:
        refinement_signals = [
            "only", "filter", "show me", "just the", "sort by",
            "most popular", "highest rated", "free ones", "cheaper",
            "more like", "similar to", "instead", "actually",
        ]
        text = user_input.lower()
        return any(sig in text for sig in refinement_signals) and bool(self._last_results)

    def _build_effective_query(self, user_input: str) -> str:
        if self._is_refinement(user_input) and self._last_query:
            return f"{self._last_query} {user_input}"
        return user_input

    # ── Main ask method ───────────────────────────────────────────────────────

    def ask(self, user_input: str) -> list[dict]:
        """Process one turn. Returns list of result dicts."""
        log.info(f"User: {user_input}")

        new_filters = self._parse_filters(user_input)
        if new_filters:
            self._active_filters.update(new_filters)

        effective_query = self._build_effective_query(user_input)

        results = retrieve(
            query=effective_query,
            qdrant_path=self.qdrant_path,
            top_k=self.top_k,
            return_k=self.return_k,
            w_semantic=self.w_semantic,
            w_category=self.w_category,
            w_popularity=self.w_popularity,
            filters=self._active_filters if self._active_filters else None,
        )

        # If hard filters returned nothing, fall back to no filter and warn
        if not results and self._active_filters:
            log.warning("Filter returned 0 results — showing unfiltered results instead")
            results = retrieve(
                query=effective_query,
                qdrant_path=self.qdrant_path,
                top_k=self.top_k,
                return_k=self.return_k,
                w_semantic=self.w_semantic,
                w_category=self.w_category,
                w_popularity=self.w_popularity,
                filters=None,
            )

        text = user_input.lower()
        if "most popular" in text or "highest upvotes" in text or "sort by upvotes" in text:
            results.sort(key=lambda x: x["upvotes"], reverse=True)

        if self.use_rag and results:
            results = add_explanations(results, effective_query)

        # Save to history
        tool_names = ", ".join(r["tool_name"] for r in results)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": f"Found {len(results)} tools: {tool_names}"})

        self._last_query   = effective_query
        self._last_results = results
        return results

    def reset(self):
        self.history = []
        self._last_query   = ""
        self._last_results = []
        self._active_filters = {}
        log.info("Conversation reset.")
