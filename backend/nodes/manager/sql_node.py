"""
Manager SQL Node
================
LangGraph node for the manager chatbot.

Handles all SQL-backed intents that require direct database retrieval.

Routing (based on ``state["intent"]``)
---------------------------------------
- ``INVENTORY``               → inventory_summary()
- ``PRODUCT_INFO``            → get_product_details(product_id) | top_selling_products()
- ``PRODUCT_REVIEW``          → get_product_reviews(product_id)
- ``ORDER_DETAILS``           → get_order_details(order_id)
- ``CUSTOMER_PURCHASE_HISTORY``→ customer_purchase_history(user_id)
- ``CUSTOMER_DETAILS``        → get_customer_details(user_id)
- ``CUSTOMER_ANALYTICS``      → handled by analytics_node (routing safety net here)
- ``SALES_ANALYTICS``         → handled by analytics_node (routing safety net here)
- ``BUSINESS_SUMMARY``        → handled by analytics_node (routing safety net here)

Entity extraction
-----------------
All entity values (product_id, order_id, user_id, keyword) come from
``state["entities"]`` populated by the intent classifier.  No regex here.

Security
--------
Managers have no ownership restrictions — they may access any order, any
customer, and any product.  Passwords are never serialised by UserService.

Debug logging
-------------
Every invocation logs: intent, entities, tool called, result key count,
and execution time.
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
from backend.schemas.manager_intent import ManagerIntent

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an intelligent retail analytics assistant for store managers at RetailHub Technologies.

You will be given structured data retrieved from the database.
Your job is to convert this data into a clear, professional management summary.

Rules:
- Do NOT make up any data. Only use what is provided.
- Format currency values in Indian Rupees (₹).
- Use tables, bullet points, or structured paragraphs for clarity.
- Highlight key insights, trends, or action items where appropriate.
- Never expose passwords, tokens, or secrets.
- Be concise, data-driven, and professional."""


def _narrate(tool_data: dict, query: str, intent: str) -> str:
    """Call the LLM to narrate structured data into a management summary."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
    data_str = json.dumps(tool_data, indent=2, default=str)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Intent: {intent}\n"
                f"Manager question: {query}\n\n"
                f"Database result:\n{data_str}"
            )
        ),
    ]
    result = llm.invoke(messages)
    return result.content  # type: ignore[return-value]


def sql_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Manager SQL retrieval node — complete implementation.

    Reads ``state["intent"]``, ``state["query"]``, and ``state["entities"]``,
    calls the appropriate manager SQL tool function(s), then uses the LLM to
    narrate the result into a professional management summary.

    Args:
        state: Current LangGraph state.

    Returns:
        Partial state dict with ``"tool_result"`` (raw JSON string) and
        ``"response"`` (LLM-narrated management summary).
    """
    t_start = time.perf_counter()
    query: str = state.get("query", "")
    intent: str = state.get("intent", ManagerIntent.GENERAL.value)
    entities: dict[str, Any] = state.get("entities", {})

    product_id: str | None = entities.get("product_id")
    order_id: str | None = entities.get("order_id")
    user_id: str | None = entities.get("user_id")

    logger.info(
        "Manager SQL node\n"
        "  Intent           : %s\n"
        "  Resolved product : %r\n"
        "  Resolved order   : %r\n"
        "  Resolved user    : %r\n"
        "  query            : %r",
        intent, product_id, order_id, user_id, query[:80],
    )

    # ------------------------------------------------------------------
    # Route to the correct tool function(s) based on intent
    # ------------------------------------------------------------------
    tool_data: dict | None = None
    tool_fn_name: str = ""

    if intent == ManagerIntent.INVENTORY.value:
        tool_fn_name = "inventory_summary"
        logger.info("Manager SQL node → inventory_summary()")
        tool_data = manager_sql_tool.inventory_summary()

    elif intent == ManagerIntent.PRODUCT_INFO.value:
        if product_id:
            tool_fn_name = f"get_product_details({product_id})"
            logger.info("Manager SQL node → get_product_details(%s)", product_id)
            tool_data = manager_sql_tool.get_product_details(product_id)
        else:
            tool_fn_name = "top_selling_products"
            logger.info("Manager SQL node → top_selling_products() (no product_id)")
            tool_data = manager_sql_tool.top_selling_products(limit=10)

    elif intent == ManagerIntent.PRODUCT_REVIEW.value:
        if product_id:
            tool_fn_name = f"get_product_reviews({product_id})"
            logger.info("Manager SQL node → get_product_reviews(%s)", product_id)
            tool_data = manager_sql_tool.get_product_reviews(product_id)
        else:
            logger.warning("Manager SQL node — PRODUCT_REVIEW but no product_id")
            return {
                "tool_result": "{}",
                "response": "Please specify a product ID (e.g. P0023) to view its reviews.",
            }

    elif intent == ManagerIntent.ORDER_DETAILS.value:
        if order_id:
            tool_fn_name = f"get_order_details({order_id})"
            logger.info("Manager SQL node → get_order_details(%s)", order_id)
            tool_data = manager_sql_tool.get_order_details(order_id)
        else:
            logger.warning("Manager SQL node — ORDER_DETAILS but no order_id")
            return {
                "tool_result": "{}",
                "response": (
                    "Please specify an order ID (e.g. ORD00123) to view its details."
                ),
            }

    elif intent == ManagerIntent.CUSTOMER_PURCHASE_HISTORY.value:
        if user_id:
            tool_fn_name = f"customer_purchase_history({user_id})"
            logger.info("Manager SQL node → customer_purchase_history(%s)", user_id)
            tool_data = manager_sql_tool.customer_purchase_history(user_id)
        else:
            logger.warning("Manager SQL node — CUSTOMER_PURCHASE_HISTORY but no user_id")
            return {
                "tool_result": "{}",
                "response": (
                    "Please specify a customer ID (e.g. U0001) to view their purchase history."
                ),
            }

    elif intent == ManagerIntent.CUSTOMER_DETAILS.value:
        if user_id:
            tool_fn_name = f"get_customer_details({user_id})"
            logger.info("Manager SQL node → get_customer_details(%s)", user_id)
            tool_data = manager_sql_tool.get_customer_details(user_id)
        else:
            logger.warning("Manager SQL node — CUSTOMER_DETAILS but no user_id")
            return {
                "tool_result": "{}",
                "response": (
                    "Please specify a customer ID (e.g. U0001) to view their profile."
                ),
            }

    # Safety nets — these intents should route to analytics_node in the graph,
    # but if they land here we still handle them correctly.
    elif intent == ManagerIntent.SALES_ANALYTICS.value:
        tool_fn_name = "full_sales_analytics"
        logger.info("Manager SQL node → full_sales_analytics() [safety net]")
        tool_data = manager_sql_tool.full_sales_analytics()

    elif intent == ManagerIntent.CUSTOMER_ANALYTICS.value:
        if user_id:
            tool_fn_name = f"customer_purchase_history({user_id})"
            profile = manager_sql_tool.get_customer_details(user_id)
            history = manager_sql_tool.customer_purchase_history(user_id)
            tool_data = {"profile": profile, "order_history": history}
        else:
            tool_fn_name = "customer_analytics"
            logger.info("Manager SQL node → customer_analytics() [safety net]")
            tool_data = manager_sql_tool.customer_analytics()

    elif intent == ManagerIntent.PRODUCT_ANALYTICS.value:
        tool_fn_name = "top_selling_products"
        logger.info("Manager SQL node → top_selling_products() [safety net]")
        tool_data = manager_sql_tool.top_selling_products(limit=10)

    elif intent == ManagerIntent.BUSINESS_SUMMARY.value:
        tool_fn_name = "business_summary_composite"
        logger.info("Manager SQL node → business_summary [safety net]")
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
    # Handle None / not-found result
    # ------------------------------------------------------------------
    if tool_data is None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.warning(
            "Manager SQL node NULL RESULT | intent=%s | tool=%s | elapsed=%.0fms",
            intent, tool_fn_name, elapsed_ms,
        )
        return {
            "tool_result": "{}",
            "response": (
                "I couldn't find the requested information. "
                "Please check the ID and try again."
            ),
        }

    # Empty dict check (for safety — service methods always return dicts)
    if not tool_data:
        return {
            "tool_result": "{}",
            "response": (
                "I wasn't able to retrieve data for that request. "
                "Please try rephrasing your question."
            ),
        }

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Manager SQL node RESULT | intent=%s | tool=%s | keys=%d | elapsed=%.0fms",
        intent, tool_fn_name, len(tool_data), elapsed_ms,
    )

    # ------------------------------------------------------------------
    # LLM narration
    # ------------------------------------------------------------------
    response = _narrate(tool_data, query, intent)
    tool_result = json.dumps(tool_data, default=str)

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Manager SQL node DONE | intent=%s | response_len=%d | total=%.0fms",
        intent, len(response), total_ms,
    )

    return {"tool_result": tool_result, "response": response}
