
from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.schemas.customer_intent import CustomerIntent, CustomerIntentOutput

load_dotenv()

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for a retail customer chatbot.

You will be given either:
  a) A plain message from the customer, OR
  b) A block labelled "Conversation so far:" showing prior turns, followed by "New message:" with the latest query.

When conversation context is provided, use it to resolve references in the new message.

Entity resolution rules:
- If the new message contains a pronoun or vague reference such as "it", "its", "this product",
  "that product", "the product", "the above product", "this item", "that item" — resolve it to
  the MOST RECENT product ID mentioned anywhere in the conversation context.
- Explicit IDs in the current message always take priority over context.
- If multiple products were discussed, use the most recently mentioned one.
- Only return null for product_id when no product can be resolved from either the message or context.

Return a JSON object with:

1. intent — EXACTLY ONE of:
   - POLICY          : questions about return, shipping, warranty, or company policies
   - PRODUCT_INFO    : questions about a product's specs, features, price, or availability
   - PRODUCT_REVIEW  : requests to see reviews or ratings for a specific product
   - PURCHASE_HISTORY: requests to see the customer's own past orders or purchases
   - ORDER_DETAILS   : questions about a specific order (status, tracking, delivery)
   - RECOMMENDATION  : requests for personalised product suggestions or recommendations
   - GENERAL         : greetings, thank-yous, or anything that does not fit above

2. product_id — resolve using the rules above. Normalise to exactly P + 4-digit zero-padded
   number (e.g. P0023). Strip extra leading zeros (P00023 → P0023). Pad short numbers (P23 → P0023).

3. order_id — the order ID (format ORD + digits, e.g. ORD00123). Extract from current message
   or resolve from context if the user refers to a previously mentioned order. Return null if none.

4. keyword — for PRODUCT_INFO queries without a product_id, the core search term
   (e.g. "laptop", "Samsung TV"). Return null for other intents or when product_id is present."""


def _normalise_product_id(raw: str | None) -> str | None:

    if not raw:
        return None
    raw = raw.strip().upper()
    match = re.match(r"^P(\d+)$", raw)
    if not match:
        return None
    digits = match.group(1).lstrip("0") or "0"
    return f"P{digits.zfill(4)}"


def _normalise_user_id(raw: str | None) -> str | None:
    """Normalise a raw user ID to ``U`` + 4-digit zero-padded number."""
    if not raw:
        return None
    raw = raw.strip().upper()
    match = re.match(r"^U(\d+)$", raw)
    if not match:
        return None
    digits = match.group(1).lstrip("0") or "0"
    return f"U{digits.zfill(4)}"


def classify_intent_node(state: dict[str, Any]) -> dict[str, Any]:

    query: str = state.get("query", "")
    conversation_context: str = state.get("conversation_context", "")

    logger.info(
        "Customer classifier START\n"
        "  Current query   : %r\n"
        "  Context present : %s",
        query[:120],
        bool(conversation_context),
    )
    logger.debug("Customer classifier — conversation context:\n%s", conversation_context)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    structured_llm = llm.with_structured_output(CustomerIntentOutput)

    # Build the user message — prepend conversation context when available
    # so the classifier can resolve pronouns/references from prior turns.
    user_message = (
        f"Conversation so far:\n{conversation_context}\n\nNew message: {query}"
        if conversation_context
        else query
    )

    result: CustomerIntentOutput = structured_llm.invoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    intent_value: str = result.intent.value

    # Normalise IDs returned by the LLM to match DB format exactly.
    entities = {
        "product_id": _normalise_product_id(result.product_id),
        "order_id": result.order_id.upper() if result.order_id else None,
        "keyword": result.keyword,
    }

    logger.info(
        "Customer classifier RESULT\n"
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
