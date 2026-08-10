"""
OpenAI embeddings factory for the RAG pipeline.

Wraps :class:`langchain_openai.OpenAIEmbeddings` so the rest of the pipeline
has a single, consistent place to obtain an embedding model instance.
"""

from __future__ import annotations

import logging
import os

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from backend.rag.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_embeddings(model: str = EMBEDDING_MODEL) -> OpenAIEmbeddings:
    """
    Return a configured :class:`~langchain_openai.OpenAIEmbeddings` instance.

    The OpenAI API key is read from the ``OPENAI_API_KEY`` environment
    variable.  When called from :mod:`backend.rag.ingest`, the variable is
    populated by ``load_dotenv`` before this function is reached.  When called
    from FastAPI, pydantic-settings has already loaded it.  The key is never
    accepted as a function parameter to avoid accidental exposure in logs or
    tracebacks.

    Args:
        model: The OpenAI embedding model name.
               Defaults to :data:`~backend.rag.config.EMBEDDING_MODEL`
               (``"text-embedding-3-small"``).

    Returns:
        A ready-to-use :class:`~langchain_openai.OpenAIEmbeddings` instance.

    Raises:
        EnvironmentError: If ``OPENAI_API_KEY`` is not set in the environment.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )
    logger.debug("Initialising OpenAI embeddings — model=%s", model)
    return OpenAIEmbeddings(model=model, api_key=SecretStr(api_key))
