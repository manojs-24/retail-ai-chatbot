"""
Manager intent classifier node.

Calls the OpenAI chat model with structured output to classify the
manager's query into a :class:`~backend.schemas.manager_intent.ManagerIntent`
and extract key entities (product_id, order_id, user_id, keyword) in a
single LLM call.

Uses ``model.with_structured_output()`` — no manual JSON parsing required.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.schemas.manager_intent import ManagerIntent, ManagerIntentOutput

load_dotenv()

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for a retail store manager chatbot.

You will be given either:
  a) A plain message from the manager, OR
  b) A block labelled "Conversation so far:" showing prior turns, followed by "New message:" with the latest query.

When conversation context is provided, use it to resolve references in the new message.

Entity resolution rules:
- If the new message contains a pronoun or vague reference such as "it", "its", "this product",
  "that product", "the product", "this item", "that item" — resolve it to the MOST RECENT
  product ID mentioned anywhere in the conversation context.
- Similarly resolve order references ("this order", "that order") and customer references
  ("this customer", "that customer") from the most recent matching entity in context.
- Explicit IDs in the current message always take priority over context.
- If multiple entities of the same type were discussed, use the most recently mentioned one.
- Only return null for an entity when it cannot be resolved from either the message or context.

Return a JSON object with:

1. intent — EXACTLY ONE of:
   - POLICY                   : questions about store policies (return, shipping, warranty, company info)
   - PRODUCT_INFO             : questions about a specific product's details, price, stock, or specs
   - PRODUCT_REVIEW           : requests to see reviews or ratings for a specific product
   - INVENTORY                : questions about stock levels, low-stock alerts, or product availability
   - ORDER_DETAILS            : requests for details of a specific order (e.g. "show order ORD00123")
   - CUSTOMER_PURCHASE_HISTORY: requests for a customer's complete purchase/order history (e.g. "what has U0023 bought")
   - CUSTOMER_DETAILS         : requests for a customer's profile information (e.g. "show details of U0023")
   - SALES_ANALYTICS          : requests for sales data, revenue trends, period comparisons, category revenue
   - CUSTOMER_ANALYTICS       : requests for customer behaviour, segmentation, highest-spending customers
   - PRODUCT_ANALYTICS        : requests for product-level performance, top-selling products list
   - BUSINESS_SUMMARY         : high-level overview or KPI dashboard of overall store performance
   - FORECAST                 : requests for predictive or ML-based sales / demand forecasting
   - GENERAL                  : greetings, thank-yous, or anything that does not fit the above

2. product_id — resolve using the rules above. Normalise to exactly P + 4-digit zero-padded
   number (e.g. P0023). Strip extra leading zeros (P00023 → P0023). Pad short numbers (P23 → P0023).

3. order_id — order ID (format ORD + digits, e.g. ORD00123). Extract from current message or
   resolve from context. Return exactly as typed, uppercased. Return null if none can be resolved.

4. user_id — customer/user ID (format U + digits, e.g. U0001). Extract from current message or
   resolve from context. Normalise: zero-pad to 4 digits (U23 → U0023). Return null if none.

5. keyword — the core search or filter term where relevant (e.g. product name, category name).
   Return null when not applicable."""


def _normalise_product_id(raw: str | None) -> str | None:
    """
    Normalise a raw product ID string to exactly ``P`` + 4-digit zero-padded number.

    Examples::

        "P00023" → "P0023"
        "P23"    → "P0023"
        "P0023"  → "P0023"
        None     → None
    """
    if not raw:
        return None
    raw = raw.strip().upper()
    match = re.match(r"^P(\d+)$", raw)
    if not match:
        return None
    digits = match.group(1).lstrip("0") or "0"
    return f"P{digits.zfill(4)}"


def _normalise_order_id(raw: str | None) -> str | None:
    """
    Normalise a raw order ID string to exactly ``ORD`` + 5-digit zero-padded number.

    Examples::

        "ORD123"   → "ORD00123"
        "ORD00123" → "ORD00123"
        None       → None
    """
    if not raw:
        return None
    raw = raw.strip().upper()
    match = re.match(r"^ORD(\d+)$", raw)
    if not match:
        return None
    digits = match.group(1).lstrip("0") or "0"
    return f"ORD{digits.zfill(5)}"


def _normalise_user_id(raw: str | None) -> str | None:
    """
    Normalise a raw user ID string to exactly ``U`` + 4-digit zero-padded number.

    Examples::

        "U001"  → "U0001"
        "U0001" → "U0001"
        "U23"   → "U0023"
        None    → None
    """
    if not raw:
        return None
    raw = raw.strip().upper()
    match = re.match(r"^U(\d+)$", raw)
    if not match:
        return None
    digits = match.group(1).lstrip("0") or "0"
    return f"U{digits.zfill(4)}"


def classify_intent_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Manager intent classifier node.

    Reads ``state["query"]``, calls the OpenAI model with structured output,
    and writes:

    - ``state["intent"]``   — the classified :class:`ManagerIntent` string value.
    - ``state["entities"]`` — dict with ``product_id``, ``order_id``,
                              ``user_id``, ``keyword`` (all normalised).

    Args:
        state: Current LangGraph state.  Must contain ``"query"``.

    Returns:
        Partial state dict with ``"intent"`` and ``"entities"``.
    """
    query: str = state.get("query", "")
    conversation_context: str = state.get("conversation_context", "")

    logger.info(
        "Manager classifier START\n"
        "  Current query   : %r\n"
        "  Context present : %s",
        query[:120],
        bool(conversation_context),
    )
    logger.debug("Manager classifier — conversation context:\n%s", conversation_context)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    structured_llm = llm.with_structured_output(ManagerIntentOutput)

    # Build the user message — prepend conversation context when available
    # so the classifier can resolve pronouns/references from prior turns.
    user_message = (
        f"Conversation so far:\n{conversation_context}\n\nNew message: {query}"
        if conversation_context
        else query
    )

    result: ManagerIntentOutput = structured_llm.invoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    intent_value: str = result.intent.value

    # Normalise all IDs returned by the LLM to match DB format exactly.
    entities = {
        "product_id": _normalise_product_id(result.product_id),
        "order_id":   _normalise_order_id(result.order_id),
        "user_id":    _normalise_user_id(result.user_id),
        "keyword":    result.keyword,
    }

    logger.info(
        "Manager classifier RESULT\n"
        "  Current query        : %r\n"
        "  Detected intent      : %s\n"
        "  Explicit product_id  : %r  →  normalised: %r\n"
        "  Resolved entities    : %s",
        query[:120],
        intent_value,
        result.product_id,
        entities["product_id"],
        entities,
    )
    return {"intent": intent_value, "entities": entities}
