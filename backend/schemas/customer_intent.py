
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CustomerIntent(str, Enum):

    POLICY = "POLICY"
    PRODUCT_INFO = "PRODUCT_INFO"
    PRODUCT_REVIEW = "PRODUCT_REVIEW"
    PURCHASE_HISTORY = "PURCHASE_HISTORY"
    ORDER_DETAILS = "ORDER_DETAILS"
    RECOMMENDATION = "RECOMMENDATION"
    GENERAL = "GENERAL"


class CustomerIntentOutput(BaseModel):

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
