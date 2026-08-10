"""
Customer intent schema.

Defines the :class:`CustomerIntent` enum and the :class:`CustomerIntentOutput`
Pydantic model used as structured output from the OpenAI intent classifier.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CustomerIntent(str, Enum):
    """
    All recognised intents for the customer chatbot.

    Values
    ------
    POLICY
        Questions about return, shipping, warranty, or company policies.
    PRODUCT_INFO
        Questions about product specifications, features, or availability.
    PRODUCT_REVIEW
        Requests to see reviews or ratings for a specific product.
    PURCHASE_HISTORY
        Requests for the customer's own past orders.
    ORDER_DETAILS
        Questions about a specific order (status, tracking, etc.).
    RECOMMENDATION
        Requests for personalised product suggestions.
    GENERAL
        Anything that does not fit a specific category (greetings, etc.).
    """

    POLICY = "POLICY"
    PRODUCT_INFO = "PRODUCT_INFO"
    PRODUCT_REVIEW = "PRODUCT_REVIEW"
    PURCHASE_HISTORY = "PURCHASE_HISTORY"
    ORDER_DETAILS = "ORDER_DETAILS"
    RECOMMENDATION = "RECOMMENDATION"
    GENERAL = "GENERAL"


class CustomerIntentOutput(BaseModel):
    """
    Structured output model returned by the customer intent classifier LLM call.

    Using a Pydantic model as the ``response_format`` guarantees the LLM
    always returns valid, typed values.

    Fields
    ------
    intent : CustomerIntent
        The single most-likely intent for the user's query.
    product_id : str | None
        Product ID extracted from the query (e.g. ``"P0023"``).
        Only populated when a product ID pattern is explicitly mentioned.
    order_id : str | None
        Order ID extracted from the query (e.g. ``"ORD00123"``).
        Only populated when an order ID pattern is explicitly mentioned.
    keyword : str | None
        The primary search keyword for product searches.
        Populated for PRODUCT_INFO queries that do not reference a product ID.
    """

    intent: CustomerIntent = Field(
        ..., description="The single most-likely intent for the user's query."
    )
    product_id: Optional[str] = Field(
        default=None,
        description=(
            "Product ID for the query (format: P followed by digits, e.g. P0023). "
            "Extract from the current message if present. "
            "If the current message uses a pronoun or reference such as 'it', 'its', "
            "'this product', 'that product', 'the product', resolve it to the most recent "
            "product ID mentioned in the conversation context. "
            "Return null only when no product can be determined from either the current "
            "message or the conversation context."
        ),
    )
    order_id: Optional[str] = Field(
        default=None,
        description=(
            "Order ID for the query (format: ORD followed by digits, e.g. ORD00123). "
            "Extract from the current message if present; otherwise resolve from conversation "
            "context if the user is referring to a previously mentioned order. "
            "Return null when no order can be determined."
        ),
    )
    keyword: Optional[str] = Field(
        default=None,
        description=(
            "The core search term for product queries (e.g. 'laptop', 'Samsung'). "
            "Return null for non-product-search intents."
        ),
    )
