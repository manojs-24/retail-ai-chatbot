"""
Manager Analytics Node
======================
LangGraph node for the manager chatbot.

Handles intents that require aggregated analytics data:
    - SALES_ANALYTICS   → full_sales_analytics() — revenue, monthly, categories, top products
    - CUSTOMER_ANALYTICS → customer_analytics()   — highest spenders, customer count
    - PRODUCT_ANALYTICS  → top_selling_products() — top 10 by units sold
    - BUSINESS_SUMMARY   → combined KPIs + inventory + top products

All data is computed by SQLAlchemy + Pandas in the service layer.
The LLM only narrates the structured result — it does NOT calculate.

Debug logging
-------------
Every invocation logs:
    - intent
    - tool function called
    - result key count
    - execution time
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.tools import manager_sql_tool

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — LLM narrates analytics data in a management style
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an intelligent retail analytics assistant for store managers at RetailHub Technologies.

You will be given structured analytics data calculated directly from the database.
Your job is to convert this into a clear, professional management summary.

Rules:
- Do NOT make up any data. Only use what is provided.
- Format all currency values in Indian Rupees (₹).
- Format large numbers: use "L" for lakhs (100,000), "Cr" for crores (10,000,000).
- Use tables, bullet points, or structured paragraphs for clarity.
- Highlight key insights, trends, and action items where appropriate.
- Clearly label any forecast values as predictions, not guarantees.
- Be concise, data-driven, and professional."""


def _narrate(tool_data: dict, query: str, intent: str) -> str:
    """Call the LLM to narrate structured analytics data."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
    data_str = json.dumps(tool_data, indent=2, default=str)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Intent: {intent}\n"
                f"Manager question: {query}\n\n"
                f"Analytics data:\n{data_str}"
            )
        ),
    ]
    result = llm.invoke(messages)
    return result.content  # type: ignore[return-value]


def analytics_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Manager analytics node — real implementation.

    Routes to the appropriate analytics tool based on ``state["intent"]``:

    - ``SALES_ANALYTICS``    → full_sales_analytics() (KPIs + monthly + categories + top products)
    - ``CUSTOMER_ANALYTICS`` → customer_analytics() (highest spenders)
    - ``PRODUCT_ANALYTICS``  → top_selling_products() (top 10 by units)
    - ``BUSINESS_SUMMARY``   → sales_summary + inventory_summary + top_selling_products

    Args:
        state: Current LangGraph state.  Must contain ``"query"`` and ``"intent"``.

    Returns:
        Partial state dict with ``"tool_result"`` (raw JSON) and
        ``"response"`` (LLM-narrated management summary).
    """
    t_start = time.perf_counter()
    query: str = state.get("query", "")
    intent: str = state.get("intent", "GENERAL")

    logger.info(
        "Analytics node START | intent=%s | query=%r", intent, query[:80]
    )

    tool_data: dict | None = None
    tool_fn_name: str = ""

    # ------------------------------------------------------------------
    # Route to the correct analytics function
    # ------------------------------------------------------------------
    if intent == "SALES_ANALYTICS":
        tool_fn_name = "full_sales_analytics"
        logger.info("Analytics node → calling full_sales_analytics()")
        tool_data = manager_sql_tool.full_sales_analytics()

    elif intent == "CUSTOMER_ANALYTICS":
        tool_fn_name = "customer_analytics"
        logger.info("Analytics node → calling customer_analytics()")
        tool_data = manager_sql_tool.customer_analytics()

    elif intent == "PRODUCT_ANALYTICS":
        tool_fn_name = "top_selling_products"
        logger.info("Analytics node → calling top_selling_products()")
        tool_data = manager_sql_tool.top_selling_products(limit=10)

    elif intent == "BUSINESS_SUMMARY":
        tool_fn_name = "business_summary_composite"
        logger.info("Analytics node → calling business_summary_composite()")
        sales = manager_sql_tool.sales_summary()
        inventory = manager_sql_tool.inventory_summary()
        top_products = manager_sql_tool.top_selling_products(limit=5)
        tool_data = {
            "sales_summary": sales,
            "inventory_summary": {
                k: v for k, v in inventory.items() if k != "low_stock_products"
            },
            "top_5_products": top_products.get("products", []),
        }

    # ------------------------------------------------------------------
    # Guard — should never be reached (routing error if it is)
    # ------------------------------------------------------------------
    if tool_data is None:
        logger.warning(
            "Analytics node — no handler for intent=%s, returning fallback", intent
        )
        return {
            "tool_result": "{}",
            "response": (
                "I wasn't able to retrieve analytics data for that request. "
                "Please try rephrasing your question."
            ),
        }

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Analytics node RESULT | intent=%s | tool=%s | keys=%d | elapsed=%.0fms",
        intent, tool_fn_name, len(tool_data), elapsed_ms,
    )

    # ------------------------------------------------------------------
    # LLM narration
    # ------------------------------------------------------------------
    response = _narrate(tool_data, query, intent)
    tool_result = json.dumps(tool_data, default=str)

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Analytics node DONE | intent=%s | response_len=%d | total=%.0fms",
        intent, len(response), total_ms,
    )

    return {"tool_result": tool_result, "response": response}
