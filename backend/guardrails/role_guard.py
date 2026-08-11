from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Intent allow-lists

# Intents a customer is permitted to trigger.
_CUSTOMER_ALLOWED_INTENTS: frozenset[str] = frozenset({
    "POLICY",
    "PRODUCT_INFO",
    "PRODUCT_REVIEW",
    "PURCHASE_HISTORY",
    "ORDER_DETAILS",
    "RECOMMENDATION",
    "GENERAL",
})

# Intents that are restricted to managers only (used for logging clarity).
_MANAGER_ONLY_INTENTS: frozenset[str] = frozenset({
    "INVENTORY",
    "ORDER_DETAILS",
    "CUSTOMER_PURCHASE_HISTORY",
    "CUSTOMER_DETAILS",
    "SALES_ANALYTICS",
    "CUSTOMER_ANALYTICS",
    "PRODUCT_ANALYTICS",
    "BUSINESS_SUMMARY",
    "FORECAST",
})

# Full set of recognised intents — union of both sets.
_ALL_KNOWN_INTENTS: frozenset[str] = (
    _CUSTOMER_ALLOWED_INTENTS | _MANAGER_ONLY_INTENTS
)


def is_customer_allowed(intent: str) -> tuple[bool, str]:

    intent_upper = intent.upper()

    if intent_upper in _CUSTOMER_ALLOWED_INTENTS:
        return True, ""

    if intent_upper in _MANAGER_ONLY_INTENTS:
        reason = (
            f"Customers are not authorised to access {intent_upper} data. "
            "Please contact your store manager for this information."
        )
        logger.warning(
            "GUARDRAIL [role_guard] role=customer | blocked intent=%s | reason=%s",
            intent_upper, reason,
        )
        return False, reason

    # Unknown intent — reject conservatively.
    reason = f"Unknown intent '{intent_upper}' — access denied."
    logger.warning(
        "GUARDRAIL [role_guard] role=customer | unknown intent=%s", intent_upper
    )
    return False, reason


def is_manager_allowed(intent: str) -> tuple[bool, str]:

    intent_upper = intent.upper()

    if intent_upper in _ALL_KNOWN_INTENTS:
        return True, ""

    reason = f"Unknown intent '{intent_upper}' — access denied."
    logger.warning(
        "GUARDRAIL [role_guard] role=manager | unknown intent=%s", intent_upper
    )
    return False, reason
