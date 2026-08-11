from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ManagerIntent(str, Enum):

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
