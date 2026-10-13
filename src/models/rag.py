"""
RAG Layer — Phase 04
=====================
For each retrieved tool, uses LangChain + Ollama (local LLM) to generate
a personalised explanation of why the tool fits the user's query.

LLM backend is controlled by .env:
  LLM_BACKEND=ollama   → uses local Ollama (free, private)
  LLM_BACKEND=openai   → uses gpt-4o-mini (requires OPENAI_API_KEY)

If Ollama is not running, explanations fall back to a template string
so the rest of the pipeline continues working.
"""

import os
import logging
from pathlib import Path

import yaml
from langchain_core.prompts import PromptTemplate

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "base_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Prompt template ───────────────────────────────────────────────────────────

EXPLANATION_PROMPT = PromptTemplate(
    input_variables=["query", "tool_name", "description", "category", "pricing"],
    template="""You are a concise AI tool recommender.

User needs: {query}

Tool: {tool_name}
Category: {category}
Pricing: {pricing}
Description: {description}

In exactly 2 sentences, explain why this tool fits the user's needs. Be specific, not generic.""",
)


# ── LLM loader ────────────────────────────────────────────────────────────────

_llm = None


def get_llm(cfg: dict):
    global _llm
    if _llm is not None:
        return _llm

    backend = os.getenv("LLM_BACKEND", cfg["llm"]["backend"]).lower()

    if backend == "ollama":
        try:
            from langchain_ollama import OllamaLLM
            _llm = OllamaLLM(
                model=os.getenv("OLLAMA_MODEL", cfg["llm"]["ollama_model"]),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=cfg["llm"]["temperature"],
                num_predict=cfg["llm"]["max_tokens"],
            )
            log.info(f"LLM: Ollama ({cfg['llm']['ollama_model']}) at {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
        except Exception as e:
            log.warning(f"Ollama init failed: {e} — using fallback template")
            _llm = None

    elif backend == "openai":
        try:
            from langchain_openai import ChatOpenAI
            _llm = ChatOpenAI(
                model=cfg["llm"]["openai_model"],
                temperature=cfg["llm"]["temperature"],
                max_tokens=cfg["llm"]["max_tokens"],
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            log.info(f"LLM: OpenAI ({cfg['llm']['openai_model']})")
        except Exception as e:
            log.warning(f"OpenAI init failed: {e} — using fallback template")
            _llm = None

    return _llm


# ── Fallback explanation ───────────────────────────────────────────────────────

def fallback_explanation(tool: dict, query: str) -> str:
    return (
        f"{tool['tool_name']} is a {tool['category'] or 'AI'} tool that matches your query. "
        f"It is {tool['pricing_tier']} and has {tool['upvotes']} upvotes from the community."
    )


# ── Single explanation ────────────────────────────────────────────────────────

def explain_tool(tool: dict, query: str, cfg: dict) -> str:
    """Generate a 2-sentence explanation for why this tool fits the query."""
    llm = get_llm(cfg)

    if llm is None:
        return fallback_explanation(tool, query)

    try:
        chain = EXPLANATION_PROMPT | llm
        response = chain.invoke({
            "query":       query,
            "tool_name":   tool["tool_name"],
            "description": tool["description"][:400],  # stay within context
            "category":    tool["category"] or "General AI",
            "pricing":     tool["pricing_tier"],
        })
        # OllamaLLM returns str; ChatOpenAI returns AIMessage
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip()
    except Exception as e:
        log.warning(f"LLM explanation failed for '{tool['tool_name']}': {e}")
        return fallback_explanation(tool, query)


# ── Batch explanation ─────────────────────────────────────────────────────────

def add_explanations(tools: list[dict], query: str) -> list[dict]:
    """
    Add an 'explanation' field to each tool dict.
    Calls the LLM once per tool (sequentially to avoid rate limits).
    """
    cfg = load_config()
    enriched = []
    for tool in tools:
        explanation = explain_tool(tool, query, cfg)
        enriched.append({**tool, "explanation": explanation})
        log.info(f"  [{tool['tool_name']}] explanation generated")
    return enriched
