# 🛒 Retail AI System

> **AI-Powered Smart Retail Intelligence & Recommendation System**

A production-ready FastAPI + Streamlit application that combines LangChain agents, LangGraph workflows, ChromaDB vector search, and OpenAI language models to deliver personalised product recommendations, intelligent customer support, and AI-driven sales analytics for retail operations.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Database** | SQLite + SQLAlchemy ORM + Alembic |
| **AI / LLM** | OpenAI GPT via LangChain & LangGraph |
| **Vector Store** | ChromaDB (persistent) |
| **Config** | pydantic-settings + python-dotenv |
| **Data** | Pandas |
| **Auth** | passlib[bcrypt] + python-jose (JWT) |
| **Package Manager** | uv |
| **Python** | 3.12+ |

---

## Project Structure

```
retail_ai/
├── backend/
│   ├── api/              # FastAPI routers (per domain)
│   ├── core/             # Config, database, logging, security
│   ├── models/           # SQLAlchemy ORM models
│   ├── schemas/          # Pydantic request/response schemas
│   ├── services/         # Business logic layer
│   ├── repositories/     # Data-access layer (repository pattern)
│   ├── agents/           # LangChain / LangGraph agent graphs
│   ├── rag/              # ChromaDB helpers and RAG pipelines
│   ├── utils/            # Shared utilities
│   ├── data/             # Sample CSV seed data
│   ├── tests/            # Pytest test suite
│   └── main.py           # FastAPI application entry point
├── frontend/
│   ├── pages/            # Streamlit multi-page modules
│   ├── components/       # Reusable UI components
│   ├── utils/            # Frontend helpers (API client, auth)
│   └── Home.py           # Streamlit entry point
├── chroma_db/            # ChromaDB persistent storage
├── database/             # SQLite database file (retail.db)
├── logs/                 # Rotating application logs
├── docs/                 # Project documentation
├── scripts/              # Start-up shell / batch scripts
├── .env.example          # Environment variable template
├── .gitignore
├── pyproject.toml        # uv-compatible project metadata
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) installed (`pip install uv` or `winget install astral-sh.uv`)

### 2. Clone and install dependencies

```bash
git clone <repo-url>
cd retail_ai
uv sync
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your values, especially OPENAI_API_KEY and SECRET_KEY
```

### 4. Start the backend

```bash
# Linux / macOS
bash scripts/start_backend.sh

# Windows
scripts\start_backend.bat

# Or directly:
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 5. Start the frontend (separate terminal)

```bash
# Linux / macOS
bash scripts/start_frontend.sh

# Windows
scripts\start_frontend.bat

# Or directly:
uv run streamlit run frontend/Home.py --server.port 8501
```

### 6. Open in browser

| Service | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Swagger UI | http://localhost:8000/docs |
| FastAPI ReDoc | http://localhost:8000/redoc |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Retail AI System` | Application display name |
| `APP_VERSION` | `1.0.0` | Semantic version |
| `DEBUG` | `False` | Enable SQLAlchemy query echo |
| `ENVIRONMENT` | `development` | Runtime environment |
| `DATABASE_URL` | `sqlite:///./database/retail.db` | SQLAlchemy DB URL |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB storage path |
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
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `GET` | `/redoc` | ReDoc UI (auto-generated) |

> Additional endpoints (auth, products, orders, recommendations) will be added in subsequent sprints.

---

## Features

### Authentication
- JWT-based login for customers and store managers
- Role-based access control (RBAC)
- Secure password hashing with bcrypt

### Customer Features
- Browse and search product catalogue
- AI-powered personalised product recommendations
- View order history and live order status
- AI chat support powered by LangChain RAG agents

### Store Manager Features
- Sales analytics dashboard with natural-language query support
- Inventory management and low-stock alerts
- Customer behaviour insights
- Policy document management (ChromaDB ingestion)

---

## Database Migrations

```bash
# Initialise Alembic (first time only)
uv run alembic init alembic

# Generate a migration after editing ORM models
uv run alembic revision --autogenerate -m "describe change"

# Apply pending migrations
uv run alembic upgrade head
```

---

## Running Tests

```bash
uv run pytest
uv run pytest -v --tb=short        # verbose with short tracebacks
uv run pytest backend/tests/       # specific directory
```

---

## Development Notes

- **Secrets**: never commit `.env`; use `.env.example` as the template.
- **Logging**: all output goes to both the console and `logs/retail_ai.log` (rotating, max 10 MB × 5 backups).
- **ChromaDB**: the persistent store lives in `chroma_db/`; this directory is git-tracked via `.gitkeep` but its contents are ignored.
- **CORS**: all origins are allowed in `development` mode; set `ENVIRONMENT=production` and configure `allow_origins` explicitly before deploying.
- **Security module** (`backend/core/security.py`): JWT and bcrypt implementation is scaffolded and ready for the auth sprint.

---

## License

MIT — see `LICENSE` for details.
