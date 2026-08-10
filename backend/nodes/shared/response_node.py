"""
Shared response node.

The final node in every graph path.  It simply reads ``state["response"]``
and returns it unchanged, acting as a clean, explicit terminal step before
``END``.

Having a dedicated response node means future cross-cutting concerns
(logging, response sanitisation, token counting, etc.) can be added here
in one place without touching individual tool nodes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def response_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Terminal node — surfaces the final response from the graph state.

    Args:
        state: The current LangGraph state dict.  Must contain ``"response"``.

    Returns:
        A partial state dict echoing the ``"response"`` key so LangGraph
        propagates it correctly to the graph output.
    """
    response: str = state.get("response", "I'm sorry, I could not generate a response.")
    logger.debug("Response node — response length=%d", len(response))
    return {"response": response}
