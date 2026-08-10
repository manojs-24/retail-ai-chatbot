"""
ChromaDB vector store helpers for the RAG pipeline.

Provides two public functions:

- :func:`build_vectorstore` — creates (or recreates) the ChromaDB collection
  from a list of document chunks and persists it to disk.
- :func:`load_vectorstore`  — loads an existing persisted collection for
  querying (used by the retriever).
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from backend.rag.config import COLLECTION_NAME, VECTOR_DB_PATH

logger = logging.getLogger(__name__)


def build_vectorstore(
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
    *,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Embed *chunks* and persist them in a new ChromaDB collection.

    If the collection already exists it is **deleted and recreated** so that
    each ingestion run produces a clean, consistent store.

    Args:
        chunks:          Chunked documents produced by :mod:`backend.rag.chunking`.
        embeddings:      Embedding model from :mod:`backend.rag.embeddings`.
        collection_name: ChromaDB collection name.
                         Defaults to :data:`~backend.rag.config.COLLECTION_NAME`.

    Returns:
        A :class:`~langchain_chroma.Chroma` instance backed by the newly
        created persistent collection.

    Raises:
        ValueError: If *chunks* is empty.
    """
    if not chunks:
        raise ValueError("No chunks provided — aborting vector store build.")

    # Ensure the persist directory exists.
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    # Delete existing collection so every ingest run starts clean.
    _delete_collection_if_exists(collection_name)

    logger.info(
        "Saving %d chunks to Chroma collection '%s' at %s ...",
        len(chunks),
        collection_name,
        VECTOR_DB_PATH,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(VECTOR_DB_PATH),
    )

    logger.info("Vector store saved successfully.")
    return vectorstore


def load_vectorstore(
    embeddings: OpenAIEmbeddings,
    *,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Load an existing persisted ChromaDB collection.

    Args:
        embeddings:      Embedding model used at ingestion time.
        collection_name: ChromaDB collection name.

    Returns:
        A :class:`~langchain_chroma.Chroma` instance connected to the
        existing persistent collection.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_PATH),
    )


def _delete_collection_if_exists(collection_name: str) -> None:
    """Delete *collection_name* from the persistent store if it exists."""
    import chromadb

    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'.", collection_name)
