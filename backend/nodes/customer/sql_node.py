"""
Customer SQL Node
=================
LangGraph node for the customer chatbot.

Executes structured database queries via the service + repository stack.

Routing (based on ``state["intent"]``)
---------------------------------------
- ``PURCHASE_HISTORY`` → get_purchase_history()
- ``ORDER_DETAILS``    → get_order_details()  (order_id from state["entities"])
- ``PRODUCT_INFO``     → get_product_details() if product_id present, else search_products(keyword)
- ``PRODUCT_REVIEW``   → get_product_reviews() (product_id from state["entities"])
- ``RECOMMENDATION``   → recommend_products()

Security
--------
``user_id`` is ALWAYS taken from ``state["user_id"]`` (the authenticated
session value).  It is never parsed from the user's message.

Entity extraction
-----------------
All entity values (product_id, order_id, keyword) come from ``state["entities"]``
which is populated by the intent classifier node.  No regex is run here.

LLM step
--------
The tool returns structured data.  The LLM's only job is to convert that
data into a friendly, natural-language response.  No SQL is ever generated
by the LLM.
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
from backend.schemas.customer_intent import CustomerIntent

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — LLM narrates structured data, never generates SQL
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a helpful retail assistant for RetailHub Technologies.

You will be given structured data retrieved from the database.
Your job is to convert this data into a clear, friendly, natural-language response.

Rules:
- Do NOT make up any data. Only use what is provided.
- Format currency values in Indian Rupees (₹).
- Use bullet points or short paragraphs for readability.
- Be concise and professional.
- If the data is empty or null, politely say no records were found."""


def _narrate(tool_data: dict, query: str, intent: str) -> str:
    """
    Send structured *tool_data* to the LLM and return a natural-language response.

    Args:
        tool_data: The dict returned by the SQL tool function.
        query:     Original user query (for context).
        intent:    The classified intent string.

    Returns:
        LLM-generated natural-language response string.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

    data_str = json.dumps(tool_data, indent=2, default=str)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Intent: {intent}\n"
                f"User question: {query}\n\n"
                f"Database result:\n{data_str}"
            )
        ),
    ]
    result = llm.invoke(messages)
    return result.content  # type: ignore[return-value]


def sql_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Customer SQL retrieval node.

    Reads ``state["intent"]``, ``state["user_id"]``, and ``state["entities"]``,
    calls the appropriate tool function, then uses the LLM to narrate the
    structured result into a natural-language response.

    Args:
        state: Current LangGraph state.  Must contain ``"query"``,
               ``"user_id"``, ``"intent"``, and ``"entities"``.

    Returns:
        Partial state dict with ``"tool_result"`` (raw JSON string) and
        ``"response"`` (LLM-narrated natural-language answer).
    """
    query: str = state.get("query", "")
    user_id: str = state.get("user_id", "")
    intent: str = state.get("intent", CustomerIntent.GENERAL.value)
    entities: dict[str, Any] = state.get("entities", {})

    product_id: str | None = entities.get("product_id")
    order_id: str | None = entities.get("order_id")
    keyword: str | None = entities.get("keyword")

    logger.info(
        "Customer SQL node\n"
        "  Intent           : %s\n"
        "  Resolved product : %r\n"
        "  Resolved order   : %r\n"
        "  Keyword          : %r\n"
        "  user_id          : %s",
        intent, product_id, order_id, keyword, user_id,
    )

    # ------------------------------------------------------------------
    # Route to the correct tool function based on intent
    # ------------------------------------------------------------------
    tool_data: dict | None = None

    if intent == CustomerIntent.PURCHASE_HISTORY.value:
        tool_data = customer_sql_tool.get_purchase_history(user_id)

    elif intent == CustomerIntent.ORDER_DETAILS.value:
        if order_id:
            tool_data = customer_sql_tool.get_order_details(order_id, user_id)
        else:
            # No order ID extracted — ask the user to clarify rather than
            # silently returning recent orders.
            return {
                "tool_result": "{}",
                "response": (
                    "I couldn't find an order ID in your message. "
                    "Please provide your order ID (e.g. ORD00123) so I can look it up for you."
                ),
            }

    elif intent == CustomerIntent.PRODUCT_INFO.value:
        if product_id:
            # Direct lookup by product ID — exact match in DB.
            tool_data = customer_sql_tool.get_product_details(product_id)
        else:
            # Keyword search — fall back to query text if no keyword extracted.
            search_term = keyword or query
            tool_data = customer_sql_tool.search_products(search_term)

    elif intent == CustomerIntent.PRODUCT_REVIEW.value:
        if product_id:
            tool_data = customer_sql_tool.get_product_reviews(product_id)
        else:
            return {
                "tool_result": "{}",
                "response": (
                    "Please specify a product ID (e.g. P0023) so I can fetch the reviews for you."
                ),
            }

    elif intent == CustomerIntent.RECOMMENDATION.value:
        tool_data = customer_sql_tool.recommend_products(user_id)

    # ------------------------------------------------------------------
    # Handle None result (product / order not found or access denied)
    # ------------------------------------------------------------------
    if tool_data is None:
        response = (
            "I couldn't find the requested information. "
            "Please check the ID and try again, or contact our support team."
        )
        return {"tool_result": "{}", "response": response}

    # ------------------------------------------------------------------
    # Handle genuinely empty-but-valid results
    # (use `is None` check above; empty dicts are valid data)
    # ------------------------------------------------------------------
    logger.info(
        "Customer SQL node — tool returned %d top-level keys", len(tool_data)
    )

    # ------------------------------------------------------------------
    # LLM narration — convert structured data to natural language
    # ------------------------------------------------------------------
    response = _narrate(tool_data, query, intent)
    tool_result = json.dumps(tool_data, default=str)

    return {"tool_result": tool_result, "response": response}
