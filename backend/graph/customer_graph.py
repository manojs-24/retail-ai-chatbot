from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.nodes.customer.classify_intent import classify_intent_node
from backend.nodes.customer.recommendation_node import recommendation_node
from backend.nodes.customer.sql_node import sql_node
from backend.nodes.shared.input_guard_node import input_guard_node
from backend.nodes.shared.output_guard_node import output_guard_node
from backend.nodes.shared.rag_node import rag_node
from backend.nodes.shared.response_node import response_node
from backend.nodes.shared.role_guard_node import role_guard_node
from backend.schemas.customer_intent import CustomerIntent
from backend.state.customer_state import CustomerState

logger = logging.getLogger(__name__)


# Guard routing helpers — pure functions, no side effects

def _route_after_input_guard(state: dict[str, Any]) -> str:
    if state.get("guard_blocked", False):
        logger.debug("Input guard blocked — routing to response_node")
        return "response_node"
    return "classify_intent"


def _route_after_role_guard(state: dict[str, Any]) -> str:

    if state.get("guard_blocked", False):
        logger.debug("Role guard blocked — routing to response_node")
        return "response_node"
    return "_route_intent"  # Sentinel — triggers the intent router below


def _route_customer_intent(state: dict[str, Any]) -> str:

    intent: str = state.get("intent", CustomerIntent.GENERAL.value)
    logger.debug("Routing customer intent → %s", intent)

    routing: dict[str, str] = {
        CustomerIntent.POLICY.value:           "rag_node",
        CustomerIntent.PRODUCT_INFO.value:     "sql_node",
        CustomerIntent.PRODUCT_REVIEW.value:   "sql_node",
        CustomerIntent.PURCHASE_HISTORY.value: "sql_node",
        CustomerIntent.ORDER_DETAILS.value:    "sql_node",
        CustomerIntent.RECOMMENDATION.value:   "recommendation_node",
        CustomerIntent.GENERAL.value:          "response_node",
    }
    return routing.get(intent, "response_node")


# Graph builder

def build_customer_graph() -> Any:
    graph = StateGraph(CustomerState)

    # Register nodes
    graph.add_node("input_guard_node",    input_guard_node)
    graph.add_node("classify_intent",     classify_intent_node)
    graph.add_node("role_guard_node",     role_guard_node)
    graph.add_node("rag_node",            rag_node)
    graph.add_node("sql_node",            sql_node)
    graph.add_node("recommendation_node", recommendation_node)
    graph.add_node("output_guard_node",   output_guard_node)
    graph.add_node("response_node",       response_node)

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

    # After role guard: blocked → response_node, clean → tool routing
    graph.add_conditional_edges(
        "role_guard_node",
        _route_customer_intent,
        {
            "rag_node":             "rag_node",
            "sql_node":             "sql_node",
            "recommendation_node":  "recommendation_node",
            "response_node":        "response_node",
        },
    )

    # Every tool node → output guard → response node
    graph.add_edge("rag_node",            "output_guard_node")
    graph.add_edge("sql_node",            "output_guard_node")
    graph.add_edge("recommendation_node", "output_guard_node")
    graph.add_edge("output_guard_node",   "response_node")

    # Terminal edge
    graph.add_edge("response_node", END)

    compiled = graph.compile()
    logger.info("Customer graph compiled successfully (with guardrails).")
    return compiled


# Module-level singleton — import and reuse across requests
customer_graph = build_customer_graph()
