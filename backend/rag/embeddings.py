
from __future__ import annotations

import logging
import os

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from backend.rag.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_embeddings(model: str = EMBEDDING_MODEL) -> OpenAIEmbeddings:

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )
    logger.debug("Initialising OpenAI embeddings — model=%s", model)
    return OpenAIEmbeddings(model=model, api_key=SecretStr(api_key))
