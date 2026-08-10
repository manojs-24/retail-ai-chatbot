"""
Manager intent schema.

Defines the :class:`ManagerIntent` enum and the :class:`ManagerIntentOutput`
Pydantic model used as structured output from the OpenAI intent classifier.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ManagerIntent(str, Enum):
    """
    All recognised intents for the manager chatbot.

    Values
    ------
    POLICY
        Questions about store policies (return, shipping, warranty, etc.).
    PRODUCT_INFO
        Questions about a specific product's details, price, or stock.
    PRODUCT_REVIEW
        Requests to see reviews or ratings for a specific product.
    INVENTORY
        Questions about current stock levels, low-stock alerts, or availability.
    ORDER_DETAILS
        Requests for details of a specific order by order ID.
    CUSTOMER_PURCHASE_HISTORY
        Requests for a specific customer's complete order history.
    CUSTOMER_DETAILS
        Requests for a specific customer's profile information.
    SALES_ANALYTICS
        Requests for sales data, revenue trends, or period comparisons.
    CUSTOMER_ANALYTICS
        Requests for customer behaviour, segmentation, or spending patterns.
    PRODUCT_ANALYTICS
        Requests for product-level performance metrics or top-selling products.
    BUSINESS_SUMMARY
        High-level overview of store performance.
    FORECAST
        Requests for predictive / ML-based sales or demand forecasting.
    GENERAL
        Anything that does not fit a specific category.
    """

    POLICY = "POLICY"
    PRODUCT_INFO = "PRODUCT_INFO"
    PRODUCT_REVIEW = "PRODUCT_REVIEW"
    INVENTORY = "INVENTORY"
    ORDER_DETAILS = "ORDER_DETAILS"
    CUSTOMER_PURCHASE_HISTORY = "CUSTOMER_PURCHASE_HISTORY"
    CUSTOMER_DETAILS = "CUSTOMER_DETAILS"
    SALES_ANALYTICS = "SALES_ANALYTICS"
    CUSTOMER_ANALYTICS = "CUSTOMER_ANALYTICS"
    PRODUCT_ANALYTICS = "PRODUCT_ANALYTICS"
    BUSINESS_SUMMARY = "BUSINESS_SUMMARY"
    FORECAST = "FORECAST"
    GENERAL = "GENERAL"


class ManagerIntentOutput(BaseModel):
    """
    Structured output model returned by the manager intent classifier LLM call.

    Using a Pydantic model as the ``response_format`` guarantees the LLM
    always returns valid, typed values.

    Fields
    ------
    intent : ManagerIntent
        The single most-likely intent for the manager's query.
    product_id : str | None
        Product ID extracted from the query (e.g. ``"P0023"``).
        Only populated when a product ID pattern is explicitly mentioned.
    order_id : str | None
        Order ID extracted from the query (e.g. ``"ORD00123"``).
        Only populated when an order ID pattern is explicitly mentioned.
    user_id : str | None
        Customer/user ID extracted from the query (e.g. ``"U0001"``).
        Only populated when a customer ID pattern is explicitly mentioned.
    keyword : str | None
        The core search or filter term where relevant.
        Return null when not applicable.
    """

    intent: ManagerIntent = Field(
        ..., description="The single most-likely intent for the manager's query."
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
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Customer ID for the query (format: U followed by digits, e.g. U0001). "
            "Extract from the current message if present; otherwise resolve from conversation "
            "context if the user is referring to a previously mentioned customer. "
            "Return null when no customer can be determined."
        ),
    )
    keyword: Optional[str] = Field(
        default=None,
        description=(
            "The core search or filter keyword where relevant (e.g. product name, category). "
            "Return null when not applicable."
        ),
    )
