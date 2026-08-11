from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.nodes.manager.analytics_node import analytics_node
from backend.nodes.manager.classify_intent import classify_intent_node
from backend.nodes.manager.forecast_node import forecast_node
from backend.nodes.manager.sql_node import sql_node
from backend.nodes.shared.input_guard_node import input_guard_node
from backend.nodes.shared.output_guard_node import output_guard_node
from backend.nodes.shared.rag_node import rag_node
from backend.nodes.shared.response_node import response_node
from backend.nodes.shared.role_guard_node import role_guard_node
from backend.schemas.manager_intent import ManagerIntent
from backend.state.manager_state import ManagerState

logger = logging.getLogger(__name__)

# Intent → node routing table
_SQL_INTENTS: frozenset[str] = frozenset({
    ManagerIntent.INVENTORY.value,
    ManagerIntent.PRODUCT_INFO.value,
    ManagerIntent.PRODUCT_REVIEW.value,
    ManagerIntent.ORDER_DETAILS.value,
    ManagerIntent.CUSTOMER_PURCHASE_HISTORY.value,
    ManagerIntent.CUSTOMER_DETAILS.value,
})

_ANALYTICS_INTENTS: frozenset[str] = frozenset({
    ManagerIntent.SALES_ANALYTICS.value,
    ManagerIntent.CUSTOMER_ANALYTICS.value,
    ManagerIntent.PRODUCT_ANALYTICS.value,
    ManagerIntent.BUSINESS_SUMMARY.value,
})


# Guard routing helpers — pure functions, no side effects

def _route_after_input_guard(state: dict[str, Any]) -> str:
    if state.get("guard_blocked", False):
        logger.debug("Input guard blocked — routing to response_node")
        return "response_node"
    return "classify_intent"


def _route_manager_intent(state: dict[str, Any]) -> str:
    if state.get("guard_blocked", False):
        logger.debug("Role guard blocked — routing to response_node")
        return "response_node"

    intent: str = state.get("intent", ManagerIntent.GENERAL.value)
    logger.debug("Routing manager intent → %s", intent)

    if intent == ManagerIntent.POLICY.value:
        return "rag_node"

    if intent in _SQL_INTENTS:
        return "sql_node"

    if intent in _ANALYTICS_INTENTS:
        return "analytics_node"

    if intent == ManagerIntent.FORECAST.value:
        return "forecast_node"

    # GENERAL and any unknown intent → respond directly
    return "response_node"


# Graph builder

def build_manager_graph() -> Any:
    graph = StateGraph(ManagerState)

    # Register nodes

    graph.add_node("input_guard_node",  input_guard_node)
    graph.add_node("classify_intent",   classify_intent_node)
    graph.add_node("role_guard_node",   role_guard_node)
    graph.add_node("rag_node",          rag_node)
    graph.add_node("sql_node",          sql_node)
    graph.add_node("analytics_node",    analytics_node)
    graph.add_node("forecast_node",     forecast_node)
    graph.add_node("output_guard_node", output_guard_node)
    graph.add_node("response_node",     response_node)


    # Entry point → input guard

    graph.add_edge(START, "input_guard_node")


    # After input guard: blocked → response_node, clean → classify_intent

    graph.add_conditional_edges(
        "input_guard_node",
        _route_after_input_guard,
        {
            "response_node":   "response_node",
            "classify_intent": "classify_intent",
        },
    )


    # After classification → role guard

    graph.add_edge("classify_intent", "role_guard_node")


    # After role guard: route to tool node or short-circuit

    graph.add_conditional_edges(
        "role_guard_node",
        _route_manager_intent,
        {
            "rag_node":       "rag_node",
            "sql_node":       "sql_node",
            "analytics_node": "analytics_node",
            "forecast_node":  "forecast_node",
            "response_node":  "response_node",
        },
    )


    # Every tool node → output guard → response node

    graph.add_edge("rag_node",          "output_guard_node")
    graph.add_edge("sql_node",          "output_guard_node")
    graph.add_edge("analytics_node",    "output_guard_node")
    graph.add_edge("forecast_node",     "output_guard_node")
    graph.add_edge("output_guard_node", "response_node")


    # Terminal edge

    graph.add_edge("response_node", END)

    compiled = graph.compile()
    logger.info("Manager graph compiled successfully (with guardrails).")
    return compiled


# Module-level singleton — import and reuse across requests
manager_graph = build_manager_graph()
