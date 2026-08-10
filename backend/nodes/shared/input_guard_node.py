"""
Input Guard Node
================
LangGraph node that runs the full input guard pipeline before the intent
classifier is invoked.

Position in graph
-----------------
    START → input_guard_node → classify_intent → …

Behaviour
---------
- Calls :func:`~backend.guardrails.input_guard.run_input_guard` with the
  raw query, role, and any already-available context.
- On **rejection**: writes ``state["guard_blocked"] = True`` and
  ``state["response"]`` with the human-readable rejection message, then
  sets ``state["intent"]`` to ``"BLOCKED"`` so the conditional router
  immediately jumps to ``response_node``.
- On **pass**: writes ``state["guard_blocked"] = False`` and continues
  normally.  Normalised entity IDs (from validation step) are written
  back into ``state["entities"]`` so downstream nodes benefit from the
  clean values.

The node does NOT call the intent classifier — it only gates access to it.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.guardrails.input_guard import run_input_guard

logger = logging.getLogger(__name__)


def input_guard_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Pre-classification input guard LangGraph node.

    Reads:
        - ``state["query"]``    — raw user input
        - ``state["role"]``     — ``"customer"`` or ``"manager"``
        - ``state["entities"]`` — optional pre-existing entities (may be empty)

    Writes:
        - ``state["guard_blocked"]``  — ``True`` if rejected, ``False`` otherwise
        - ``state["response"]``       — rejection message (only when blocked)
        - ``state["intent"]``         — ``"BLOCKED"`` (only when blocked)
        - ``state["entities"]``       — normalised entity values (when passed)

    Args:
        state: Current LangGraph state dict.

    Returns:
        Partial state update dict.
    """
    query: str = state.get("query", "")
    role: str = state.get("role", "customer")
    entities: dict[str, Any] = state.get("entities", {})

    logger.info(
        "Input guard node — role=%s query_len=%d", role, len(query)
    )

    # Intent is not yet known at this stage (classifier hasn't run).
    # Entity validation will still run if entities dict is pre-populated.
    result = run_input_guard(
        query=query,
        role=role,
        intent="",       # Not yet classified
        entities=entities,
    )

    if not result.allowed:
        logger.warning(
            "Input guard BLOCKED | role=%s | reason=%s", role, result.rejection_reason
        )
        return {
            "guard_blocked": True,
            "response": result.rejection_reason,
            "intent": "BLOCKED",
        }

    logger.debug("Input guard PASSED | role=%s | checks=%s", role, result.guard_checks)
    return {
        "guard_blocked": False,
        "entities": result.normalised_entities,
    }
