from __future__ import annotations

import logging
from typing import Any

from backend.guardrails.role_guard import is_customer_allowed, is_manager_allowed

logger = logging.getLogger(__name__)


def role_guard_node(state: dict[str, Any]) -> dict[str, Any]:

    intent: str = state.get("intent", "GENERAL")
    role: str = state.get("role", "customer")

    # If a previous guard already blocked the request, pass through immediately.
    if state.get("guard_blocked", False):
        return {}

    logger.info("Role guard node — role=%s intent=%s", role, intent)

    if role == "customer":
        allowed, reason = is_customer_allowed(intent)
    else:
        allowed, reason = is_manager_allowed(intent)

    if not allowed:
        logger.warning(
            "Role guard BLOCKED | role=%s | intent=%s | reason=%s",
            role, intent, reason,
        )
        return {
            "guard_blocked": True,
            "response": reason,
            "intent": "BLOCKED",
        }

    logger.debug("Role guard PASSED | role=%s | intent=%s", role, intent)
    return {"guard_blocked": False}
