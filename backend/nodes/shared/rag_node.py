from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.rag.retriever import get_retriever

# Load .env so the node works whether called from a script or FastAPI.
load_dotenv()

logger = logging.getLogger(__name__)

# System prompt — strictly grounds the LLM in the retrieved context.
_SYSTEM_PROMPT = """You are a helpful retail assistant for RetailHub Technologies.
Answer the user's question ONLY using the context provided below.
If the answer is not found in the context, respond with:
"I'm sorry, I don't have information about that in our knowledge base. \
Please contact our support team for further assistance."

Do NOT make up information. Do NOT use outside knowledge.
Be concise, friendly, and professional."""


def rag_node(state: dict[str, Any]) -> dict[str, Any]:

    query: str = state.get("query", "")
    logger.info("RAG node — query=%r", query[:80])

    # 1. Retrieve relevant chunks from ChromaDB.
    retriever = get_retriever()
    docs = retriever.invoke(query)

    # Serialise documents for storage in state (LangGraph state must be
    # JSON-serialisable to support persistence / checkpointing).
    serialised_docs: list[dict[str, Any]] = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    logger.info("RAG node — retrieved %d document chunks", len(serialised_docs))

    # 2. Build a grounded context string from the retrieved chunks.
    if serialised_docs:
        context_parts: list[str] = []
        for i, doc in enumerate(serialised_docs, start=1):
            source = doc["metadata"].get("document_name", "Unknown")
            page = doc["metadata"].get("page", "?")
            context_parts.append(
                f"[Source {i}: {source}, page {page}]\n{doc['page_content']}"
            )
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = ""

    # 3. Call the LLM to generate a grounded answer.
    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )
        ),
    ]

    response_message = llm.invoke(messages)
    response: str = response_message.content  # type: ignore[assignment]
    logger.info("RAG node — response generated (%d chars)", len(response))

    return {
        "retrieved_documents": serialised_docs,
        "response": response,
    }
