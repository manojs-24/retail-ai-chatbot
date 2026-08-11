# Retail AI System

> **AI-Powered Smart Retail Intelligence & Recommendation System**

A production-ready FastAPI + Streamlit application that combines LangGraph agentic workflows, LangChain RAG pipelines, ChromaDB vector search, scikit-learn ML models, and OpenAI language models to deliver personalised product recommendations, intelligent customer and manager chat, sales forecasting, and AI-driven analytics for retail operations.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                        │
│  Home · Login · CustomerDashboard · ManagerDashboard            │
│  customer_chat · manager_chat · Validation (eval dashboard)      │
│                    Sarvam AI  STT / TTS                          │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP (REST / JWT)
┌────────────────────────▼────────────────────────────────────────┐
│                       FastAPI Backend                            │
│  /auth  ·  /health  ·  (per-domain routers)                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  LangGraph Agent Graphs                   │   │
│  │                                                          │   │
│  │   Customer Graph              Manager Graph              │   │
│  │   ─────────────               ────────────               │   │
│  │   input_guard_node            input_guard_node           │   │
│  │        ↓                           ↓                     │   │
│  │   classify_intent             classify_intent            │   │
│  │        ↓                           ↓                     │   │
│  │   role_guard_node             role_guard_node            │   │
│  │        ↓ (conditional)             ↓ (conditional)       │   │
│  │   ┌────────────────┐         ┌─────────────────────┐    │   │
│  │   │ rag_node       │         │ rag_node             │    │   │
│  │   │ sql_node       │         │ sql_node             │    │   │
│  │   │ recommendation │         │ analytics_node       │    │   │
│  │   │ _node          │         │ forecast_node        │    │   │
│  │   └───────┬────────┘         └──────────┬──────────┘    │   │
│  │           └──────────┬───────────────────┘               │   │
│  │                      ↓                                    │   │
│  │              output_guard_node                            │   │
│  │                      ↓                                    │   │
│  │               response_node → END                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  RAG Pipeline  │  │  ML Modules  │  │   Guardrails     │    │
│  │                │  │              │  │                  │    │
│  │  loader        │  │  sales_      │  │  input_guard     │    │
│  │  chunking ①   │  │  forecast    │  │  output_guard    │    │
│  │  embeddings    │  │  demand_     │  │  role_guard      │    │
│  │  vectorstore   │  │  prediction  │  │  injection_guard │    │
│  │  retriever ②  │  │  customer_   │  │  validation_     │    │
│  │  ingest        │  │  segmenta-   │  │  guard           │    │
│  └───────┬────────┘  │  tion        │  └──────────────────┘    │
│          │           │  sentiment_  │                           │
│          │           │  analysis    │  ┌──────────────────┐    │
│  ┌───────▼────────┐  │  product_    │  │  Evaluation      │    │
│  │   ChromaDB     │  │  performance │  │                  │    │
│  │  (persistent)  │  │  inventory_  │  │  rag_evaluation  │    │
│  └────────────────┘  │  prediction  │  │  sql_evaluation  │    │
│                      └──────┬───────┘  │  ml_evaluation   │    │
│  ┌───────────────────────────▼──────┐  └──────────────────┘    │
│  │  SQLite + SQLAlchemy ORM         │                           │
│  │  Users · Products · Orders       │                           │
│  │  OrderItems · Reviews            │                           │
│  └──────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘

① Chunking fallback: RecursiveCharacterTextSplitter → manual char-window slicer
② Retrieval fallback: ChromaDB vector search → BM25 keyword retriever (rank-bm25)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI 0.115+ · Uvicorn |
| **Frontend** | Streamlit 1.40+ |
| **Agent Orchestration** | LangGraph 0.2+ |
| **LLM / Chains** | LangChain 0.3+ · OpenAI GPT-4o-mini (bot) · GPT-4o (eval judge) |
| **RAG** | LangChain · ChromaDB (persistent) · OpenAI `text-embedding-3-small` |
| **RAG Fallbacks** | Manual char-window chunker · BM25 (rank-bm25) |
| **ML** | scikit-learn (Polynomial Regression, Random Forest, KMeans) |
| **Database** | SQLite · SQLAlchemy ORM · Alembic |
| **Voice** | Sarvam AI (STT · TTS · translation) |
| **Evaluation** | RAGAS 0.2 (RAG) · custom SQL intent evaluator · scikit-learn metrics (ML) |
| **Auth** | JWT (python-jose) · bcrypt (passlib) |
| **Config** | pydantic-settings · python-dotenv |
| **Package Manager** | uv |
| **Python** | 3.12+ |

---

## Project Structure

```
retail_ai/
├── backend/
│   ├── api/                    # FastAPI routers
│   │   └── auth.py             #   POST /auth/login, /auth/logout
│   ├── core/                   # App config, DB engine, logging, security
│   ├── models/                 # SQLAlchemy ORM models (User, Product, Order …)
│   ├── schemas/                # Pydantic request/response + intent enums
│   ├── services/               # Business logic (auth, orders, analytics …)
│   ├── repositories/           # Data-access layer (repository pattern)
│   ├── graph/
│   │   ├── customer_graph.py   # LangGraph customer agent (compiled singleton)
│   │   └── manager_graph.py    # LangGraph manager agent (compiled singleton)
│   ├── nodes/
│   │   ├── shared/             # input_guard · output_guard · role_guard
│   │   │                       # rag_node · response_node
│   │   ├── customer/           # classify_intent · sql_node · recommendation_node
│   │   └── manager/            # classify_intent · sql_node · analytics_node
│   │                           # forecast_node
│   ├── guardrails/             # injection_guard · input_guard · output_guard
│   │                           # role_guard · validation_guard
│   ├── rag/
│   │   ├── config.py           # Chunk size, overlap, model constants
│   │   ├── loader.py           # PyPDFLoader — loads policy PDFs
│   │   ├── chunking.py         # RecursiveCharacterTextSplitter + manual fallback
│   │   ├── embeddings.py       # OpenAI embeddings factory
│   │   ├── vectorstore.py      # ChromaDB build / load
│   │   ├── retriever.py        # Vector retriever + BM25 fallback
│   │   ├── ingest.py           # End-to-end ingestion pipeline CLI
│   │   └── chroma.py           # Low-level ChromaDB client factory
│   ├── ml/
│   │   ├── sales_forecast.py       # Polynomial Regression — 30-day revenue forecast
│   │   ├── demand_prediction.py    # Random Forest — product demand prediction
│   │   ├── inventory_prediction.py # Inventory restock prediction
│   │   ├── customer_segmentation.py# KMeans customer segmentation
│   │   ├── product_performance.py  # Product performance scoring
│   │   └── sentiment_analysis.py   # Review sentiment analysis
│   ├── tools/
│   │   ├── customer_sql_tool.py    # NL→SQL tool (customer domain)
│   │   └── manager_sql_tool.py     # NL→SQL tool (manager domain)
│   ├── evaluation/
│   │   ├── rag_evaluation.py   # RAGAS pipeline (bot=gpt-4o-mini, judge=gpt-4o)
│   │   ├── sql_evaluation.py   # SQL intent accuracy evaluator
│   │   └── ml_evaluation.py    # MAE / RMSE / accuracy metrics for ML models
│   ├── state/                  # LangGraph state schemas (CustomerState, ManagerState)
│   ├── data/
│   │   ├── policy-data/        # 6 policy PDFs (Return, Shipping, Warranty …)
│   │   └── *.csv               # Seed data (users, products, orders, reviews)
│   ├── database/               # DB init + seeding scripts
│   ├── tests/
│   │   ├── rag-evaluation/     # rag_test_cases.json (12 Q&A pairs)
│   │   ├── sql-evaluation/     # sql_test_cases.json
│   │   └── smoke_*.py          # Smoke tests
│   └── main.py                 # FastAPI entry point
├── frontend/
│   ├── Home.py                 # Streamlit entry point
│   ├── pages/
│   │   ├── Login.py            # Auth page
│   │   ├── CustomerDashboard.py# Customer overview + product browse
│   │   ├── customer_chat.py    # Customer AI chat (voice + text)
│   │   ├── ManagerDashboard.py # Manager analytics dashboard
│   │   ├── manager_chat.py     # Manager AI chat (voice + text)
│   │   └── Validation.py       # RAG · SQL · ML evaluation dashboard
│   └── utils/
│       ├── auth.py             # Session helpers, require_login guard
│       ├── chat_memory.py      # In-session conversation history
│       └── sarvam.py           # Sarvam AI: STT · TTS · translation
├── database/                   # SQLite retail.db (git-ignored)
├── logs/                       # Rotating app logs (git-ignored)
├── scripts/                    # start_backend / start_frontend shell scripts
├── .env.example                # Environment variable template
├── pyproject.toml              # uv project + dependency manifest
└── uv.lock                     # Locked dependency graph
```

---

## Features

### Customer Agent
- **Intent classification** across 7 intents: `POLICY`, `PRODUCT_INFO`, `PRODUCT_REVIEW`, `PURCHASE_HISTORY`, `ORDER_DETAILS`, `RECOMMENDATION`, `GENERAL`
- **Policy Q&A** via RAG (ChromaDB + OpenAI embeddings) over 6 policy documents
- **Order & product queries** via NL→SQL tool against live database
- **AI product recommendations** based on purchase history and preferences
- **Voice input / output** via Sarvam AI (STT + TTS + multilingual translation)

### Manager Agent
- **Intent classification** across 13 intents including `SALES_ANALYTICS`, `FORECAST`, `INVENTORY`, `CUSTOMER_ANALYTICS`, `PRODUCT_ANALYTICS`, `BUSINESS_SUMMARY`
- **Policy Q&A** via the same shared RAG pipeline
- **Operational queries** via NL→SQL (inventory, orders, customer details)
- **Analytics node**: revenue trends, top products, customer behaviour insights
- **Forecast node**: 30-day sales forecast using Polynomial Regression
- **Voice support** via Sarvam AI

### RAG Pipeline
- PDF ingestion from `backend/data/policy-data/` (6 documents)
- `RecursiveCharacterTextSplitter` (700 chars / 200 overlap) with metadata enrichment
- **Chunking fallback**: pure-Python char-window slicer when splitter raises
- OpenAI `text-embedding-3-small` embeddings → ChromaDB persistent store
- **Retrieval fallback**: BM25 keyword retriever (rank-bm25) when vector store is unavailable
- Fallback chunks carry audit metadata (`fallback_chunking`, `bm25_fallback`)

### ML Models
| Model | Algorithm | Output |
|---|---|---|
| Sales Forecast | Polynomial Regression (deg 2) | 30-day revenue + trend |
| Demand Prediction | Random Forest Regressor | Predicted units demand |
| Inventory Prediction | Rule-based + ML | Restock recommendations |
| Customer Segmentation | KMeans clustering | Customer segments |
| Product Performance | Composite scoring | Performance tiers |
| Sentiment Analysis | TextBlob / rule-based | Positive / Neutral / Negative |

### Guardrail System
Every message passes through a 7-check pipeline before reaching any agent node:

| # | Check | Blocks |
|---|---|---|
| 1 | Empty query | Blank / whitespace input |
| 2 | Length check | > 2,000 characters |
| 3 | Prompt injection | Jailbreak / system-override patterns |
| 4 | SQL injection | DROP, UNION, `--` comment patterns |
| 5 | Off-topic filter | Politics, sports, medical, finance … |
| 6 | Role-based intent | Customers accessing manager-only intents |
| 7 | Entity validation | Malformed IDs (order, product, user) |

Output guard additionally screens LLM responses before they reach the user.

### Evaluation Dashboard (`Validation.py`)
- **RAG evaluation**: RAGAS 0.2 metrics — Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness. Bot model: `gpt-4o-mini`. Judge model: `gpt-4o`.
- **SQL evaluation**: Intent classification accuracy + route correctness across customer and manager test cases.
- **ML evaluation**: MAE, RMSE, accuracy metrics for all 6 ML models.

### Authentication
- JWT-based login for customers and store managers
- Role-based access control (RBAC) enforced at both API and agent layers
- Secure password hashing with bcrypt
- Session management via Streamlit session state

---

## Quick Start

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

```bash
pip install uv
# or
winget install astral-sh.uv
```

### 1. Clone and install

```bash
git clone <repo-url>
cd retail_ai
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
OPENAI_API_KEY=sk-...
SARVAM_API_KEY=...       # optional — only required for voice features
```

### 3. Initialise the database

```bash
uv run python backend/database/init_db.py
uv run python backend/database/seed_database.py
```

### 4. Ingest policy documents into ChromaDB

```bash
uv run python backend/rag/ingest.py
```

### 5. Start the backend

```bash
# Windows
scripts\start_backend.bat

# Linux / macOS
bash scripts/start_backend.sh

# Or directly
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 6. Start the frontend (separate terminal)

```bash
# Windows
scripts\start_frontend.bat

# Linux / macOS
bash scripts/start_frontend.sh

# Or directly
uv run streamlit run frontend/Home.py --server.port 8501
```

### 7. Open in browser

| Service | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Swagger UI | http://localhost:8000/docs |
| FastAPI ReDoc | http://localhost:8000/redoc |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `SARVAM_API_KEY` | *(optional)* | Sarvam AI key for voice features |
| `APP_NAME` | `Retail AI System` | Application display name |
| `APP_VERSION` | `1.0.0` | Semantic version |
| `DEBUG` | `False` | Enable SQLAlchemy query echo |
| `ENVIRONMENT` | `development` | Runtime environment |
| `DATABASE_URL` | `sqlite:///./database/retail.db` | SQLAlchemy DB URL |
| `CHROMA_DB_PATH` | `./backend/vector_db/chroma_db` | ChromaDB storage path |
| `SECRET_KEY` | *(required)* | JWT HMAC secret |
| `ALGORITHM` | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | JWT TTL in minutes |
| `LOG_LEVEL` | `INFO` | Root log level |
| `LOG_FILE_PATH` | `./logs/retail_ai.log` | Rotating log file path |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Liveness ping |
| `GET` | `/health` | Structured health check |
| `POST` | `/auth/login` | JWT login (email + password) |
| `POST` | `/auth/logout` | Server-side logout acknowledgement |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

---

## Running Tests & Evaluation

```bash
# Unit / smoke tests
uv run pytest
uv run pytest -v --tb=short

# RAG evaluation (requires ChromaDB populated + OPENAI_API_KEY)
uv run python -c "from backend.evaluation.rag_evaluation import run_rag_evaluation; run_rag_evaluation()"

# Or use the Validation page in the Streamlit UI
```

---

## Database Migrations

```bash
# Generate a migration after editing ORM models
uv run alembic revision --autogenerate -m "describe change"

# Apply pending migrations
uv run alembic upgrade head
```

---

## Development Notes

- **Secrets**: never commit `.env`; use `.env.example` as the template.
- **Logging**: structured output to console and `logs/retail_ai.log` (rotating, max 10 MB × 5 backups).
- **ChromaDB**: persistent store at `backend/vector_db/chroma_db/`; contents are git-ignored.
- **RAG fallbacks**: if the LangChain splitter fails, a pure-Python char-window fallback activates automatically. If ChromaDB / OpenAI is unavailable at query time, a BM25 keyword retriever activates. Both paths are transparent to callers.
- **CORS**: all origins allowed in `development`; set `ENVIRONMENT=production` and configure `allow_origins` before deploying.
- **Agent models**: all production nodes use `gpt-4o-mini`. The RAGAS evaluation judge uses `gpt-4o` for more reliable metric scores.

---

## License

MIT — see `LICENSE` for details.
