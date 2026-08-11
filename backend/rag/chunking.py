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

    stem = Path(filename).stem.lower()
    for key, label in _DOCUMENT_TYPE_MAP.items():
        if key in stem:
            return label
    return "Policy Document"


def _manual_chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Pure-Python character-window fallback used when RecursiveCharacterTextSplitter raises.

    Slices each document's ``page_content`` into fixed-size windows with the
    requested overlap.  No third-party splitter is involved so this path cannot
    itself fail due to library issues.

    Chunks produced here carry ``metadata["fallback_chunking"] = True`` so
    callers / monitoring can identify them.
    """
    chunks: list[Document] = []
    idx = 0
    step = max(1, chunk_size - chunk_overlap)

    for doc in documents:
        text: str = doc.page_content or ""
        if not text.strip():
            continue  # skip entirely blank pages

        start = 0
        while start < len(text):
            chunk_text = text[start : start + chunk_size]
            source: str = doc.metadata.get("source", "")
            filename: str = Path(source).name
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "document_name": filename,
                        "document_type": _infer_document_type(filename),
                        "chunk_id": idx,
                        "fallback_chunking": True,
                    },
                )
            )
            idx += 1
            start += step

    logger.warning(
        "Manual fallback chunking produced %d chunks from %d pages.",
        len(chunks),
        len(documents),
    )
    return chunks


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split *documents* into chunks using RecursiveCharacterTextSplitter.

    If the splitter raises for any reason (e.g. ``None`` page content, bad
    separator pattern), the call transparently falls back to
    :func:`_manual_chunk_documents` which is a pure-Python character-window
    splitter with no external dependencies.
    """
    try:
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

    except Exception as exc:
        logger.warning(
            "RecursiveCharacterTextSplitter failed (%s) — "
            "falling back to manual chunking.",
            exc,
        )
        return _manual_chunk_documents(documents, chunk_size, chunk_overlap)
