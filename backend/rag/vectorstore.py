
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

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_PATH),
    )


def _delete_collection_if_exists(collection_name: str) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'.", collection_name)
