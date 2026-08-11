from __future__ import annotations

import logging

import chromadb
from chromadb import Collection, PersistentClient

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Well-known collection names
COLLECTION_COMPANY_POLICIES = "company_policies"
COLLECTION_PRODUCT_CATALOG = "product_catalog"

# Module-level client cache (one client per process is sufficient for SQLite-
# backed ChromaDB; replace with a connection pool if switching to a remote
# chroma server).
_client: PersistentClient | None = None


def get_chroma_client() -> PersistentClient:

    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        logger.info("ChromaDB client initialised — path=%s", settings.CHROMA_DB_PATH)
    return _client


def get_or_create_collection(name: str) -> Collection:

    client = get_chroma_client()
    collection = client.get_or_create_collection(name=name)
    logger.debug("Collection ready: %s", name)
    return collection


def init_chroma() -> None:

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
