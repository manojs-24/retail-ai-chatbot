
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from backend.rag.config import DATA_PATH

logger = logging.getLogger(__name__)


def load_pdfs(data_path: Path = DATA_PATH) -> list[Document]:

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_path}"
        )

    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {data_path}")

    logger.info("Loading PDFs from %s ...", data_path)

    documents: list[Document] = []
    for pdf_path in pdf_files:
        logger.debug("  Loading: %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        documents.extend(pages)
        logger.debug("    → %d pages", len(pages))

    logger.info("Loaded %d PDF(s) → %d pages total", len(pdf_files), len(documents))
    return documents
