"""
RAG pipeline configuration.

All constants used by the ingestion and retrieval pipeline are defined here.
Import from this module — never scatter magic numbers across pipeline files.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths (resolved relative to this file so the pipeline works regardless
# of the working directory from which ingest.py is executed)
# ---------------------------------------------------------------------------

# backend/
_BACKEND_DIR: Path = Path(__file__).resolve().parent.parent

# backend/data/policy-data/
DATA_PATH: Path = _BACKEND_DIR / "data" / "policy-data"

# backend/vector_db/chroma_db/
VECTOR_DB_PATH: Path = _BACKEND_DIR / "vector_db" / "chroma_db"

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

COLLECTION_NAME: str = "retail_rag"

# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

# Policy documents tend to have long paragraphs — 1000 / 200 gives good
# context coverage without exceeding typical embedding token windows.
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# Separators ordered from coarsest to finest granularity.
CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

TOP_K: int = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
