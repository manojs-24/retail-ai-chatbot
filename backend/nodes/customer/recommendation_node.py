"""
Customer Recommendation Node
=============================
LangGraph node for product recommendations in the customer chatbot.

Delegates to :func:`~backend.tools.customer_sql_tool.recommend_products`
(currently top-selling products as the Phase 1 baseline) and uses the
LLM to narrate the results in a friendly, personalised way.

The node is kept separate from the SQL node so that future phases can
swap in a full recommendation agent (collaborative filtering, LangGraph
sub-graph, etc.) without touching the SQL retrieval logic.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.tools import customer_sql_tool

load_dotenv()

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful retail assistant for RetailHub Technologies.

You will be given a list of product recommendations retrieved from our database.
Your job is to present them in an engaging, helpful way.

Rules:
- Highlight the top 3–5 products most prominently.
- Mention product name, brand, category, price, and units sold.
- Format prices in Indian Rupees (₹).
- Be friendly and encouraging.
- Do NOT make up any products or prices not present in the data."""


def recommendation_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Customer recommendation node.

    Retrieves top-selling product recommendations for the authenticated
    customer and uses the LLM to present them conversationally.

    Args:
        state: Current LangGraph state.  Must contain ``"query"``
               and ``"user_id"``.

    Returns:
        Partial state dict with ``"tool_result"`` (raw JSON) and
        ``"response"`` (LLM-narrated recommendation message).
    """
    query: str = state.get("query", "")
    user_id: str = state.get("user_id", "")

    logger.info(
        "Recommendation node — user_id=%s query=%r", user_id, query[:60]
    )

    # ------------------------------------------------------------------
    # Retrieve recommendations via the tool layer
    # ------------------------------------------------------------------
    tool_data = customer_sql_tool.recommend_products(user_id)
    logger.info(
        "Recommendation node — retrieved %d recommendations",
        tool_data.get("count", 0),
    )

    # ------------------------------------------------------------------
    # LLM narration
    # ------------------------------------------------------------------
    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)

    data_str = json.dumps(tool_data, indent=2, default=str)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User request: {query}\n\n"
                f"Recommended products data:\n{data_str}"
            )
        ),
    ]
    result = llm.invoke(messages)
    response: str = result.content  # type: ignore[assignment]

    return {
        "tool_result": json.dumps(tool_data, default=str),
        "response": response,
    }
