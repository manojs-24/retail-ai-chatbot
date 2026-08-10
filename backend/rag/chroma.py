"""
ChromaDB client and collection helpers.

Provides:
- ``get_chroma_client()`` — returns a persistent ChromaDB client.
- ``get_or_create_collection(name)`` — returns (or creates) a named collection.
- ``init_chroma()`` — called at application startup to pre-create well-known
  collections and verify the vector store is reachable.
"""

from __future__ import annotations

import logging

import chromadb
from chromadb import Collection, PersistentClient

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ---------------------------------------------------------------------------
# Well-known collection names
# ---------------------------------------------------------------------------
COLLECTION_COMPANY_POLICIES = "company_policies"
COLLECTION_PRODUCT_CATALOG = "product_catalog"

# Module-level client cache (one client per process is sufficient for SQLite-
# backed ChromaDB; replace with a connection pool if switching to a remote
# chroma server).
_client: PersistentClient | None = None


def get_chroma_client() -> PersistentClient:
    """
    Return the module-level :class:`chromadb.PersistentClient` instance.

    The client is created on first call and reused on subsequent calls
    (lazy singleton pattern).

    Returns:
        A connected :class:`chromadb.PersistentClient`.
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        logger.info("ChromaDB client initialised — path=%s", settings.CHROMA_DB_PATH)
    return _client


def get_or_create_collection(name: str) -> Collection:
    """
    Return an existing ChromaDB collection or create it if it does not exist.

    Args:
        name: The collection name (must be a valid ChromaDB identifier).

    Returns:
        The requested :class:`chromadb.Collection`.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=name)
    logger.debug("Collection ready: %s", name)
    return collection


def init_chroma() -> None:
    """
    Initialise well-known ChromaDB collections at application startup.

    Creates the following collections if they do not already exist:

    - ``company_policies`` — stores embedded company policy documents for RAG.
    - ``product_catalog``  — stores embedded product descriptions for semantic search.

    Raises:
        Exception: Re-raises any ChromaDB connectivity or filesystem errors.
    """
    try:
        get_or_create_collection(COLLECTION_COMPANY_POLICIES)
        get_or_create_collection(COLLECTION_PRODUCT_CATALOG)
        logger.info(
            "ChromaDB ready — collections: %s, %s",
            COLLECTION_COMPANY_POLICIES,
            COLLECTION_PRODUCT_CATALOG,
        )
    except Exception:
        logger.exception("Failed to initialise ChromaDB collections")
        raise
