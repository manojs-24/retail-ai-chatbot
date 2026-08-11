
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly
# (e.g. `python backend/rag/ingest.py` from the retail_ai/ directory).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before any module that reads environment variables is imported.
# dotenv is a no-op when the variables are already set in the environment,
# so this is safe to call unconditionally.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from backend.rag.config import LOG_FORMAT  # noqa: E402 — path setup must come first
from backend.rag.chunking import split_documents  # noqa: E402
from backend.rag.embeddings import get_embeddings  # noqa: E402
from backend.rag.loader import load_pdfs  # noqa: E402
from backend.rag.vectorstore import build_vectorstore  # noqa: E402

# Logging — configure root logger for the ingestion run
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_ingestion() -> None:

    start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Retail AI — RAG Ingestion Pipeline")
    logger.info("=" * 60)

    # Step 1 — Load PDFs
    logger.info("Step 1/4 — Loading PDFs...")
    documents = load_pdfs()

    # Step 2 — Chunk documents
    logger.info("Step 2/4 — Splitting into chunks...")
    chunks = split_documents(documents)

    # Step 3 — Initialise embeddings
    logger.info("Step 3/4 — Generating embeddings (OpenAI)...")
    embeddings = get_embeddings()

    # Step 4 — Build and persist vector store
    logger.info("Step 4/4 — Saving to Chroma...")
    build_vectorstore(chunks, embeddings)

    elapsed = time.perf_counter() - start
    logger.info("=" * 60)
    logger.info("Done. Ingestion completed in %.1f seconds.", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
