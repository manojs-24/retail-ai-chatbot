"""
Role Guard
==========
Enforces role-based access control at the *intent* level.

Rules
-----
**Customer** may only access intents that concern their own account:
    POLICY, PRODUCT_INFO, PRODUCT_REVIEW, PURCHASE_HISTORY,
    ORDER_DETAILS, RECOMMENDATION, GENERAL

**Manager** may access all intents — no restrictions.

The guard operates on the intent *string* value (post-classification) so it
integrates cleanly with both the input guard (early rejection) and with the
role-guard LangGraph node (post-classification gating).

Design principles
-----------------
- Pure functions — no I/O, no state mutation, no LLM calls.
- Returns ``(allowed: bool, reason: str)`` tuples for consistent handling.
- Easily extended: add/remove intents from the frozensets below.
- Unit-test friendly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent allow-lists
# ---------------------------------------------------------------------------

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
    """
    Check whether a customer is permitted to trigger *intent*.

    Args:
        intent: The classified intent string (e.g. ``"POLICY"``).

    Returns:
        ``(True, "")`` if the intent is allowed for customers.
        ``(False, reason)`` if the intent is restricted.

    Example::

        allowed, reason = is_customer_allowed("INVENTORY")
        # allowed=False, reason="Customers are not authorised to access INVENTORY data."
    """
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
    """
    Check whether a manager is permitted to trigger *intent*.

    Managers have access to all known intents.  Only truly unknown intent
    strings are rejected (as a safety net against classifier bugs).

    Args:
        intent: The classified intent string (e.g. ``"SALES_ANALYTICS"``).

    Returns:
        ``(True, "")`` always for known intents.
        ``(False, reason)`` for unrecognised intent strings.

    Example::

        allowed, reason = is_manager_allowed("SALES_ANALYTICS")
        # allowed=True, reason=""
    """
    intent_upper = intent.upper()

    if intent_upper in _ALL_KNOWN_INTENTS:
        return True, ""

    reason = f"Unknown intent '{intent_upper}' — access denied."
    logger.warning(
        "GUARDRAIL [role_guard] role=manager | unknown intent=%s", intent_upper
    )
    return False, reason
