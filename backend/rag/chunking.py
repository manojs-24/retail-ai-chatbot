"""
Text chunking for the RAG ingestion pipeline.

Splits raw page-level :class:`~langchain_core.documents.Document` objects
into smaller, overlapping chunks suitable for embedding.  Each chunk is
enriched with normalised metadata before being returned.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
)

logger = logging.getLogger(__name__)

# Mapping of lowercase stem → human-readable document type label.
_DOCUMENT_TYPE_MAP: dict[str, str] = {
    "return policy": "Return Policy",
    "warranty policy": "Warranty Policy",
    "shipping policy": "Shipping Policy",
    "frequently asked questions": "FAQ",
    "faqs": "FAQ",
    "product manuals": "Product Manual",
    "company information": "Company Information",
}


def _infer_document_type(filename: str) -> str:
    """
    Derive a human-readable document type from a PDF filename stem.

    Args:
        filename: The PDF filename (with or without extension).

    Returns:
        A human-readable document type string, or ``"Policy Document"``
        if no mapping is found.
    """
    stem = Path(filename).stem.lower()
    for key, label in _DOCUMENT_TYPE_MAP.items():
        if key in stem:
            return label
    return "Policy Document"


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split a list of page-level documents into smaller overlapping chunks.

    Each chunk retains the original ``source`` and ``page`` metadata from
    its parent document, and gains three additional fields:

    - ``document_name`` — filename (without directory path).
    - ``document_type`` — inferred human-readable category.
    - ``chunk_id``      — sequential integer identifier within the batch.

    Args:
        documents:     Raw page-level documents from :mod:`backend.rag.loader`.
        chunk_size:    Maximum character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        A list of enriched, chunked :class:`~langchain_core.documents.Document`
        objects ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
        length_function=len,
    )

    chunks: list[Document] = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        source: str = chunk.metadata.get("source", "")
        filename: str = Path(source).name
        chunk.metadata["document_name"] = filename
        chunk.metadata["document_type"] = _infer_document_type(filename)
        chunk.metadata["chunk_id"] = idx

    logger.info("Created %d chunks from %d pages", len(chunks), len(documents))
    return chunks
