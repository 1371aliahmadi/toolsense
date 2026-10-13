<h1 align="center">🔍 ToolSense</h1>

<p align="center">
  <strong>Hybrid AI Tool Discovery Engine</strong><br>
  Find the right AI tool from 2,121 indexed tools — in plain English.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-DC143C?style=flat"/>
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white"/>
</p>

---

## Overview

**ToolSense** is a semantic AI tool recommender built as a university project (BHT Berlin, Semester 2). It lets you describe what you need in plain English and returns the most relevant AI tools from a curated index of 2,121 tools scraped from ProductHunt, HuggingFace, AwesomeList, FutureTools, and Futurepedia.

The system combines **SBERT semantic embeddings** with **BM25 keyword matching** and a **gradient category scorer** into a single hybrid ranking pipeline — with weights tuned by Optuna and monitored for drift with Evidently.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ToolSense Pipeline                     │
│                                                             │
│  Sources         Processing          Serving                │
│  ──────          ──────────          ───────                │
│  ProductHunt  →  Scrape (Ph01)   →  FastAPI  ←─── Streamlit│
│  HuggingFace  →  Clean (Ph02)   →   /search       UI       │
│  AwesomeList  →  Embed (Ph03)   →  Qdrant                  │
│  FutureTools  →  Tune  (Ph05)   →  SBERT                   │
│  Futurepedia  →  Eval  (Ph06)   →  BM25                    │
│                  Annotate(Ph08) →  RAG (LLM)               │
└─────────────────────────────────────────────────────────────┘
```

### Hybrid Ranking Formula

```
final_score = w_semantic × SBERT_cosine
            + w_bm25     × BM25_Okapi
            + w_category × gradient_category_score
            + w_pop      × log_normalised_upvotes
```

**Optuna-tuned weights:**

| Signal | Weight |
|--------|--------|
| SBERT semantic (all-mpnet-base-v2, 768-dim) | 0.41 |
| BM25 keyword (rank-bm25, Okapi) | 0.45 |
| Category boost (gradient adjacency graph) | 0.14 |
| Popularity (log-normalised upvotes) | 0.00 |

### Dynamic Weight Shifting

The ranker automatically shifts weights based on query type, without any user input:

| Query Type | Detection | Strategy |
|------------|-----------|----------|
| **Named entity** | Mid-sentence capitalised word (e.g. *"alternatives to Jasper"*) | BM25 dominates → precision for exact tool names |
| **Short** | ≤3 meaningful tokens after stopword removal | BM25 boosted → keyword precision |
| **Conversational** | Long natural-language phrase | SBERT boosted → semantic intent understanding |

---

## Features

- 🔎 **Hybrid search** — SBERT + BM25 + category scoring, tuned by Optuna
- 🧠 **RAG explanations** — per-result LLM explanations via Ollama or OpenAI
- ⚡ **Semantic cache** — Redis-backed query cache with cosine similarity deduplication
- ✍️ **Spell correction** — domain-aware typo correction (whitelists AI/tech acronyms)
- 👍 **Session feedback** — thumbs up/down adjusts category weights in real time
- 📊 **Drift monitoring** — Evidently reports on query log distribution over time
- 🐳 **Docker-ready** — single `docker compose up` starts API + UI
- ☸️ **Kubernetes manifests** — `k8s/` includes deployment and service YAML
- 📈 **WandB experiment tracking** — all Optuna runs and evaluation metrics logged

---

## Project Structure

```
toolsense/
├── app/
│   └── streamlit_app.py        # Streamlit UI (search, feedback, history)
├── src/
│   ├── api/main.py             # FastAPI backend (search, feedback, cache, monitoring)
│   ├── models/
│   │   ├── ranker.py           # Hybrid ranker (SBERT + BM25 + category + popularity)
│   │   ├── rag.py              # RAG layer (LangChain + Ollama/OpenAI explanations)
│   │   └── conversation.py     # Multi-turn conversation context
│   ├── data_pipeline/          # Scrapers (ProductHunt, HuggingFace, AwesomeList, ...)
│   ├── embeddings/generator.py # SBERT embedding + Qdrant indexing
│   ├── evaluation/             # Precision@K, NDCG, perturbation tests
│   ├── annotation/             # LLM + BM25 ground-truth annotators
│   ├── cache/semantic_cache.py # Redis semantic cache
│   └── monitoring/             # Evidently drift monitor + query logger
├── config/
│   ├── base_config.yaml        # All hyperparameters in one place
│   └── best_weights.yaml       # Optuna-tuned ranking weights
├── data/                       # Raw/clean CSVs, Qdrant store, annotation exports
├── k8s/                        # Kubernetes deployment + service manifests
├── run_phase01.py              # Scrape
├── run_phase02.py              # Clean & validate
├── run_phase03.py              # Embed & index
├── run_phase05.py              # Tune weights (Optuna)
├── run_phase06.py              # Evaluate
├── run_phase08.py              # Ground-truth annotation pipeline
├── run_phase09.py              # Start API + UI
├── Dockerfile
├── docker-compose.yml
└── dvc.yaml                    # DVC pipeline (scrape → clean → embed → tune → eval)
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/1371aliahmadi/toolsense.git
cd toolsense
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_BACKEND, OPENAI_API_KEY (optional), REDIS_HOST, etc.
```

### 3. Run the pipeline (first time only)

```bash
python run_phase01.py   # scrape ~2100 tools
python run_phase02.py   # clean & validate
python run_phase03.py   # embed + build Qdrant index
```

> **Skip this** if `data/clean_tools.csv` and `data/qdrant/` already exist.

### 4. Start the app

```bash
python run_phase09.py
```

This launches:
- **FastAPI** at `http://localhost:8000` (API docs at `/docs`)
- **Streamlit UI** at `http://localhost:8501`

---

## Docker

```bash
docker compose up --build
```

| Service | Port |
|---------|------|
| FastAPI API | `http://localhost:8000` |
| Streamlit UI | `http://localhost:8501` |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/weights` | Current Optuna-tuned weights |
| `POST` | `/search` | Hybrid tool search |
| `POST` | `/feedback` | Submit thumbs up/down |
| `GET` | `/cache/stats` | Redis cache statistics |
| `DELETE` | `/cache` | Flush cache |
| `GET` | `/monitor/summary` | Query log summary |
| `POST` | `/monitor/report` | Generate Evidently drift report |

**Example search request:**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "generate AI videos for YouTube", "top_k": 5}'
```

---

## Configuration

All parameters live in [`config/base_config.yaml`](config/base_config.yaml):

```yaml
embeddings:
  active_model: "model_b"        # all-mpnet-base-v2 (768-dim, higher MTEB score)

ranker:
  w_semantic: 0.60
  w_category: 0.25
  top_k: 10                      # ANN candidates from Qdrant
  return_k: 5                    # final results returned

llm:
  backend: "ollama"              # "ollama" or "openai"
  ollama_model: "llama3.2:3b"

optuna:
  n_trials: 100
  metric: "precision_at_10"
```

Override any value via `.env` — see `.env.example` for all supported variables.

---

## Pipeline Stages (DVC)

```
dvc repro           # run full pipeline end-to-end
dvc dag             # visualise DAG
```

| Stage | Script | Output |
|-------|--------|--------|
| `scrape` | `run_phase01.py` | `data/raw_tools.csv` |
| `clean` | `run_phase02.py` | `data/clean_tools.csv` |
| `embed` | `run_phase03.py` | `data/qdrant/` |
| `tune` | `run_phase05.py` | `config/best_weights.yaml` |
| `evaluate` | `run_phase06.py` | `data/eval_metrics.json` |
| `annotate` | LLM annotator | `data/annotations_export.json` |
| `ground_truth` | `run_phase08.py` | `data/ground_truth.json` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers` (all-mpnet-base-v2) |
| Vector DB | Qdrant (local HNSW index) |
| Keyword search | `rank-bm25` (Okapi BM25) |
| LLM / RAG | LangChain + Ollama (llama3.2:3b) or OpenAI (gpt-4o-mini) |
| Hyperparameter tuning | Optuna (100 trials) |
| Experiment tracking | Weights & Biases (WandB) |
| Data quality | Great Expectations, ydata-profiling |
| Drift monitoring | Evidently |
| Semantic cache | Redis |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Containerisation | Docker + Docker Compose |
| Orchestration | Kubernetes |
| Pipeline versioning | DVC |

---

## Data Sources

| Source | Tools indexed | Notes |
|--------|--------------|-------|
| ProductHunt | ~700 | Top AI products by upvotes |
| HuggingFace Spaces | ~600 | Top spaces by likes |
| AwesomeList | ~400 | Curated AI tool lists |
| FutureTools | ~250 | AI tool directory |
| Futurepedia | ~170 | AI tool database |

**Total: 2,121 tools** after deduplication and quality filtering.

---

## License

This project was developed as part of the BHT Berlin Data Science programme (Semester 2, 2026). For educational use.
