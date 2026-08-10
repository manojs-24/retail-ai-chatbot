"""
Role Guard Node
===============
LangGraph node that enforces role-based access control *after* the intent
classifier has run and written ``state["intent"]``.

Position in graph
-----------------
    classify_intent → role_guard_node → conditional_routing → …

Behaviour
---------
- Calls the appropriate :func:`is_customer_allowed` or
  :func:`is_manager_allowed` function based on ``state["role"]``.
- On **rejection**: writes ``state["guard_blocked"] = True``,
  ``state["response"]`` with the human-readable reason, and sets
  ``state["intent"]`` to ``"BLOCKED"`` so the conditional router skips
  all tool nodes and goes straight to ``response_node``.
- On **pass**: writes ``state["guard_blocked"] = False`` and lets the
  graph continue to the conditional routing step.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.guardrails.role_guard import is_customer_allowed, is_manager_allowed

logger = logging.getLogger(__name__)


def role_guard_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Post-classification role guard LangGraph node.

    Reads:
        - ``state["intent"]``  — the classified intent string
        - ``state["role"]``    — ``"customer"`` or ``"manager"``

    Writes:
        - ``state["guard_blocked"]`` — ``True`` if rejected
        - ``state["response"]``      — rejection message (only when blocked)
        - ``state["intent"]``        — set to ``"BLOCKED"`` when rejected

    Args:
        state: Current LangGraph state dict.

    Returns:
        Partial state update dict.
    """
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
