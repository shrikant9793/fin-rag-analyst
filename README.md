# 📊 Multi-Agent RAG — Financial Market Research Analyst

> LangGraph · Qdrant Hybrid Search · Gemini/Groq/Ollama · NeMo Guardrails · Langfuse · Ragas · Streamlit · Docker

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Graph                    │
│                                                             │
│  [Financial Data Fetcher]                                   │
│         │                                                   │
│         ▼                                                   │
│  [RAG Retriever Agent] ◄── Qdrant Hybrid Search             │
│         │              (BGE-M3 dense + BM42 sparse + RRF)  │
│         ▼                                                   │
│  [Compliance / Guardrail Agent] ◄── NeMo Guardrails         │
│         │                                                   │
│         ▼                                                   │
│  [Report Synthesizer Agent]                                 │
│         │                                                   │
│         ▼                                                   │
│  ★ HITL Node — PAUSE ★ ──► Analyst reviews in Streamlit UI  │
│         │   (Approve / Override)                            │
│         ▼                                                   │
│  [Final Report Compiler]                                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
  Streamlit UI Output + Langfuse Trace
```

---

## Quick Start — Day 1 (Infra + RAG Ingestion)

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- A free API key: [Google AI Studio](https://aistudio.google.com/app/apikey) **or** [Groq](https://console.groq.com/keys)

---

### Step 1 — Clone & Configure

```bash
git clone https://github.com/your-org/fin-rag-analyst.git
cd fin-rag-analyst

# Copy env template
cp .env.example .env

# Edit .env — add your API key for the active LLM provider
nano .env
```

Edit `config/config.yaml` to set your active LLM provider:
```yaml
llm:
  active: "gemini"   # change to "groq" or "ollama"
```

---

### Step 2 — Start Infrastructure (Qdrant + Langfuse)

```bash
# Start Qdrant and Langfuse (Day 1 only — no app container yet)
docker compose -f docker/docker-compose.yml up qdrant langfuse postgres -d

# Verify Qdrant is healthy
curl http://localhost:6333/healthz
# → {"title":"qdrant - the vector search engine","version":"..."}

# Verify Langfuse is up
open http://localhost:3000     # Create your account here
```

---

### Step 3 — Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4 — Ingest Financial Documents

Put your SEC filings or earnings transcripts in `data/raw/`:

```bash
# Ingest a single PDF
python scripts/ingest.py \
  --path data/raw/AAPL_10K_2024.pdf \
  --ticker AAPL \
  --doc-type sec_filing \
  --filing-date 2024-10-31 \
  --company "Apple Inc."

# Ingest all files in a folder
python scripts/ingest.py \
  --path data/raw/ \
  --ticker MSFT \
  --doc-type earnings_transcript \
  --filing-date 2024-07-30
```

---

### Step 5 — Verify Retrieval

```bash
# Smoke test — returns top 5 chunks for a sample query
python scripts/test_retrieval.py \
  --query "What is Apple's gross margin for Q4 2024?" \
  --ticker AAPL \
  --top-n 5
```

Expected output:
```
✓ Qdrant connected
  Collection : fin_documents
  Points     : 847
  Status     : green

Top 5 Retrieved Chunks
┌───┬───────────┬────────┬───────────────────┬────────────┬──────────────────────────┐
│ # │ RRF Score │ Ticker │ Doc Type          │ Date       │ Content Preview          │
├───┼───────────┼────────┼───────────────────┼────────────┼──────────────────────────┤
│ 1 │ 0.0317    │ AAPL   │ earnings_transcr… │ 2024-10-31 │ Gross margin: 46.2%...   │
...
✓ Retrieval test passed — 5 chunks returned
```

---

### Step 6 — Run Unit Tests

```bash
# Unit tests (no Qdrant required)
pytest tests/test_rag_pipeline.py -v

# Integration tests (requires running Qdrant)
QDRANT_URL=http://localhost:6333 pytest tests/test_rag_pipeline.py -v -m integration
```

---

## Ollama Local Setup (No API Key Needed)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh   # Linux/Mac
# Windows: https://ollama.ai/download

# Pull Llama 3.1 8B (4.7 GB)
ollama pull llama3.1:8b

# Verify
ollama run llama3.1:8b "Summarise Apple Q4 2024 earnings in 3 bullet points."
```

Then switch in `config/config.yaml`:
```yaml
llm:
  active: "ollama"
```

---

## Langfuse Observability Setup

1. Navigate to `http://localhost:3000` after running `docker compose up`
2. Create an account (self-hosted, no cloud required)
3. Go to **Settings → API Keys** → create a new key pair
4. Copy keys into your `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```
5. Every LLM call will now appear in the Langfuse dashboard with:
   - Per-agent latency waterfall
   - Token usage and cost per run
   - Input/output logging per node

---

## Project Structure

```
fin-rag-analyst/
├── src/
│   ├── __init__.py
│   ├── config.py            # Settings loader (pydantic-settings + yaml)
│   ├── llm_factory.py       # LLM provider switcher (Gemini/Groq/Ollama)
│   ├── rag_pipeline.py      # Hybrid RAG: ingest + retrieve
│   ├── state.py             # LangGraph AgentState (Day 2)
│   ├── graph.py             # LangGraph agent graph + HITL (Day 2)
│   ├── observability.py     # Langfuse trace helpers (Day 3)
│   └── agents/
│       ├── fetcher.py       # Financial Data Fetcher Agent (Day 2)
│       ├── retriever.py     # RAG Retriever Agent (Day 2)
│       ├── compliance.py    # Guardrail / Compliance Agent (Day 3)
│       └── synthesizer.py   # Report Synthesizer Agent (Day 2)
├── config/
│   ├── config.yaml          # Central config (LLM, Qdrant, Langfuse, Eval)
│   └── guardrails/          # NeMo Guardrails colang rules (Day 3)
├── tests/
│   ├── test_rag_pipeline.py # Day 1 unit + integration tests
│   ├── eval_suite.py        # Ragas + DeepEval (Day 5)
│   └── golden_dataset.json  # Ground-truth QA pairs (Day 5)
├── docker/
│   ├── Dockerfile           # Multi-stage Python 3.11 build
│   └── docker-compose.yml   # Full stack: app + qdrant + langfuse + postgres
├── ui/
│   ├── app.py               # Streamlit entry point (Day 4)
│   └── components/
│       ├── hitl_panel.py    # Analyst approval UI (Day 4)
│       └── eval_dashboard.py
├── scripts/
│   ├── ingest.py            # CLI document ingestion
│   └── test_retrieval.py    # Retrieval smoke test
├── data/
│   ├── raw/                 # Drop financial docs here before ingesting
│   └── processed/
├── .github/
│   └── workflows/
│       └── cicd.yml         # GitHub Actions CI/CD (Day 5)
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Day-by-Day Progress

| Day | Status | Focus |
|-----|--------|-------|
| **1** | ✅ Complete | Repo, config, Qdrant hybrid RAG, Docker infra |
| **2** | 🔲 Pending  | LangGraph agents, HITL pause/resume |
| **3** | 🔲 Pending  | NeMo Guardrails, Langfuse tracing |
| **4** | 🔲 Pending  | Streamlit UI, full Docker compose |
| **5** | 🔲 Pending  | Ragas + DeepEval eval suite, GitHub Actions CI/CD |
