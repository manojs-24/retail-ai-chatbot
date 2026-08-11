from __future__ import annotations

import logging

from langchain_core.retrievers import BaseRetriever

from backend.rag.config import COLLECTION_NAME, TOP_K
from backend.rag.embeddings import get_embeddings
from backend.rag.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


def _get_bm25_fallback_retriever(top_k: int) -> BaseRetriever:
    """Build a BM25 keyword retriever directly from the on-disk PDFs.

    This path has **no network dependency** — it loads the raw PDF files,
    splits them (with the manual chunking fallback if needed), and constructs
    a ``BM25Retriever`` (backed by ``rank_bm25``) entirely in-process.

    The returned retriever implements :class:`langchain_core.retrievers.BaseRetriever`
    so it is a transparent drop-in for the vector-store retriever.

    Chunks retrieved via this path carry ``metadata["bm25_fallback"] = True``
    to aid observability / monitoring.
    """
    from langchain_community.retrievers import BM25Retriever

    from backend.rag.chunking import split_documents
    from backend.rag.loader import load_pdfs

    logger.warning(
        "AI/vector retrieval unavailable — activating BM25 keyword fallback "
        "(top_k=%d).",
        top_k,
    )

    documents = load_pdfs()
    chunks = split_documents(documents)

    # Tag every chunk so callers can tell they came from the fallback path.
    for chunk in chunks:
        chunk.metadata["bm25_fallback"] = True

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = top_k

    logger.info(
        "BM25 fallback retriever ready — %d chunks indexed.", len(chunks)
    )
    return retriever


def get_retriever(
    *,
    top_k: int = TOP_K,
    collection_name: str = COLLECTION_NAME,
) -> BaseRetriever:
    """Return the primary vector-store retriever, or a BM25 fallback.

    Primary path
    ------------
    Loads OpenAI embeddings and the persisted ChromaDB collection, then
    returns a ``VectorStoreRetriever`` configured for cosine-similarity search.

    Fallback path
    -------------
    If *any* step of the primary path raises (missing API key, network error,
    corrupt / missing Chroma DB), the exception is caught, a ``WARNING`` is
    emitted, and :func:`_get_bm25_fallback_retriever` is returned instead.
    The BM25 retriever needs only the PDF files on disk — no external services.

    If the fallback itself raises (e.g. no PDFs on disk), the exception
    propagates naturally so the caller receives a clear error rather than
    silent empty results.
    """
    try:
        logger.debug(
            "Loading retriever — collection=%s, top_k=%d", collection_name, top_k
        )
        embeddings = get_embeddings()
        vectorstore = load_vectorstore(embeddings, collection_name=collection_name)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
        logger.debug("Vector store retriever ready.")
        return retriever

    except Exception as exc:
        logger.warning(
            "Vector store retriever failed (%s) — falling back to BM25.", exc
        )
        return _get_bm25_fallback_retriever(top_k)
