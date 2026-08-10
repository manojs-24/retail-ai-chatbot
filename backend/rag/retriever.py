"""
LangChain retriever for the RAG pipeline.

Exposes :func:`get_retriever`, which returns a LangChain
:class:`~langchain_core.retrievers.BaseRetriever` backed by the persisted
ChromaDB vector store.

Usage::

    from backend.rag.retriever import get_retriever

    retriever = get_retriever()
    docs = retriever.invoke("What is the return policy for damaged products?")
    for doc in docs:
        print(doc.metadata["document_name"], "— page", doc.metadata["page"])
        print(doc.page_content[:300])
        print()
"""

from __future__ import annotations

import logging

from langchain_core.retrievers import BaseRetriever

from backend.rag.config import COLLECTION_NAME, TOP_K
from backend.rag.embeddings import get_embeddings
from backend.rag.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


def get_retriever(
    *,
    top_k: int = TOP_K,
    collection_name: str = COLLECTION_NAME,
) -> BaseRetriever:
    """
    Return a LangChain retriever over the persisted ChromaDB collection.

    The retriever uses **similarity search** to find the *top_k* most
    semantically relevant chunks for a given query string.

    The vector store must have been populated by running
    :mod:`backend.rag.ingest` at least once before calling this function.

    Args:
        top_k:           Number of chunks to return per query.
                         Defaults to :data:`~backend.rag.config.TOP_K` (5).
        collection_name: ChromaDB collection to query.
                         Defaults to :data:`~backend.rag.config.COLLECTION_NAME`.

    Returns:
        A :class:`~langchain_core.retrievers.BaseRetriever` ready for
        ``.invoke()`` calls.

    Raises:
        Exception: If the ChromaDB collection does not exist (ingestion has
                   not been run yet).
    """
    logger.debug(
        "Loading retriever — collection=%s, top_k=%d", collection_name, top_k
    )
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings, collection_name=collection_name)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    logger.debug("Retriever ready.")
    return retriever
