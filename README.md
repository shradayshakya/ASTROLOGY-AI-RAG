# Jyotish AI (Vedic-RAG)

An Agentic RAG Streamlit application that grounds Vedic astrology responses in two sources:

- Accurate astronomical data from FreeAstrologyAPI
- Classical interpretive rules from Brihat Parashara Hora Shastra (BPHS) via Pinecone

The app collects DOB, TOB, and City, fetches the correct divisional chart (D1/D9/D10), retrieves relevant BPHS passages, and synthesizes the answer. All SVG charts are rendered directly in the chat.

## Architecture

- Orchestration: LangChain Agent (ReAct) + Tools
- Frontend: Streamlit `main.py`
- LLM Factory: OpenAI / Google Gemini / AWS Bedrock (via env `LLM_PROVIDER`)
- Vector DB: Pinecone (Serverless)
- Cache & History: MongoDB Atlas (`api_cache`, `chat_history`)
- Observability: LangSmith (Tracing enabled)

See detailed design in [docs/PRD.md](docs/PRD.md) and [docs/EDD.md](docs/EDD.md).

## Project Layout

```
vedic-rag/
├── .env.example
├── Dockerfile
├── requirements.txt
├── main.py
├── scripts/
│   ├── ingest.py            # Ingest BPHS PDF into Pinecone
│   └── setup_prompts.py     # Push system prompt to LangChain Hub
├── src/
│   ├── config.py            # Env + config
│   ├── llm_factory.py       # Factory Method for LLMs
│   ├── utils.py             # Geocoding + timezone offset
│   ├── vector_store.py      # Pinecone retriever helper
│   ├── tools.py             # D1/D9/D10 tools + MongoDB caching + BPHS search
│   └── agent.py             # AgentExecutor with tools + chat history
├── data/
│   └── BPHS.pdf             # Source PDF (example path)
└── docs/
    ├── PRD.md
    └── EDD.md
```

## Setup

### 1) Python Environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2) Environment Variables

Copy `.env.example` → `.env` and fill values:

- `LLM_PROVIDER`: `openai` | `google_genai` | `bedrock`
- `OPENAI_API_KEY` / `GOOGLE_API_KEY` / AWS creds
- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
- `MONGO_URI`, `MONGO_DB_NAME`, `MONGO_API_CACHE_COLLECTION`, `MONGO_CHAT_HISTORY_COLLECTION`
- `FREE_ASTROLOGY_API_USER_ID`, `FREE_ASTROLOGY_API_KEY`
- `LANGCHAIN_API_KEY` (LangSmith tracing)

### 3) Ingest BPHS into Pinecone

Ensure your BPHS PDF exists at `data/BPHS.pdf` (or update the path in [scripts/ingest.py](scripts/ingest.py)).

```bash
python scripts/ingest.py
```

This creates/uses the Pinecone serverless index and loads embedded chunks.

### 4) Run the App

```bash
streamlit run main.py
```

Enter Email (Session ID), DOB, TOB, and City. Ask questions about career (D10), marriage (D9), or health (D1).

### 5) VS Code Debugging

Use [ .vscode/launch.json ] to debug via the Python `streamlit` module. Set breakpoints in [src/agent.py](src/agent.py) and tools.

## LLM Factory

Select provider via `LLM_PROVIDER` in `.env`:

- `openai` → GPT-4o
- `google_genai` → Gemini 1.5 Pro
- `bedrock` → Claude 3 Sonnet (AWS Bedrock)

## Tools & Caching

Tools in [src/tools.py](src/tools.py):

- `get_d10_chart(dob, tob, city)` – career (D10)
- `get_d9_chart(dob, tob, city)` – marriage (D9)
- `get_d1_chart(dob, tob, city)` – general health (D1)
- `search_bphs(query)` – search BPHS via Pinecone

MongoDB caching keys: `dob+tob+lat+lon+chart_type`. Checks `api_cache` collection before calling FreeAstrologyAPI.

## Docker

Build and run with env:

```bash
docker build -t jyotish-ai:latest .
docker run --rm -p 8501:8501 --env-file .env jyotish-ai:latest
```

Streamlit will be available at http://localhost:8501.

## Notes

- SVG charts must be rendered with `st.markdown(svg, unsafe_allow_html=True)`.
- Prompts should be managed via LangChain Hub; push with [scripts/setup_prompts.py](scripts/setup_prompts.py) and pull in [src/agent.py](src/agent.py).
- For production, consider Auth0 for OAuth and deployment on Streamlit Cloud or AWS.
