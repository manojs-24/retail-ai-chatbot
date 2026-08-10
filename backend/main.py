"""
FastAPI application entry point for the Retail AI backend.

Run locally with::

    uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Logging must be the very first import so all subsequent loggers inherit the
# configured handlers.
from backend.core.logging import get_logger, setup_logging
from backend.core.config import get_settings
from backend.core.database import init_db
from backend.rag.chroma import init_chroma
from backend.api.auth import router as auth_router

settings = get_settings()

# ---------------------------------------------------------------------------
# Bootstrap logging immediately (before the app object is created so that
# uvicorn's own startup messages are captured too).
# ---------------------------------------------------------------------------
setup_logging(log_level=settings.LOG_LEVEL, log_file_path=settings.LOG_FILE_PATH)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan context manager (replaces deprecated @app.on_event handlers)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Handle startup and shutdown tasks."""
    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    logger.info(
        "Starting %s v%s — environment=%s",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
    )
    init_db()
    init_chroma()
    logger.info("Application startup complete.")

    yield  # Application runs here.

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    logger.info("Shutting down %s.", settings.APP_NAME)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI-Powered Smart Retail Intelligence & Recommendation System. "
        "Provides product recommendations, order management, and AI-driven "
        "customer insights via LangChain / LangGraph agents."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS middleware
# (Allow all origins in development; tighten allow_origins in production.)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(auth_router)

# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"], summary="Root ping")
async def root() -> dict[str, str]:
    """Return a simple liveness message."""
    return {"message": "Retail AI Backend Running"}


@app.get("/health", tags=["Health"], summary="Detailed health check")
async def health_check() -> dict[str, Any]:
    """
    Return structured health information.

    Response fields:
    - **status**: Always ``"healthy"`` if the process is alive.
    - **app**: Application display name.
    - **version**: Current semantic version.
    - **environment**: Runtime environment name.
    """
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }
