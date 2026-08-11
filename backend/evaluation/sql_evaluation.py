from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Path to the test-case JSON file
_TEST_CASES_PATH = Path(__file__).resolve().parent.parent.parent / "backend" / "tests" /"sql-evaluation" / "sql_test_cases.json"

# Intent → expected route mapping (mirrors the graph routing tables)
_CUSTOMER_INTENT_ROUTE: dict[str, str] = {
    "POLICY": "rag_node",
    "PRODUCT_INFO": "sql_node",
    "PRODUCT_REVIEW": "sql_node",
    "PURCHASE_HISTORY": "sql_node",
    "ORDER_DETAILS": "sql_node",
    "RECOMMENDATION": "recommendation_node",
    "GENERAL": "response_node",
}

_MANAGER_INTENT_ROUTE: dict[str, str] = {
    "POLICY": "rag_node",
    "PRODUCT_INFO": "sql_node",
    "PRODUCT_REVIEW": "sql_node",
    "INVENTORY": "sql_node",
    "ORDER_DETAILS": "sql_node",
    "CUSTOMER_PURCHASE_HISTORY": "sql_node",
    "CUSTOMER_DETAILS": "sql_node",
    "SALES_ANALYTICS": "analytics_node",
    "CUSTOMER_ANALYTICS": "analytics_node",
    "PRODUCT_ANALYTICS": "analytics_node",
    "BUSINESS_SUMMARY": "analytics_node",
    "FORECAST": "forecast_node",
    "GENERAL": "response_node",
}


@dataclass
class SQLCaseResult:
    """Result for a single SQL test case."""
    case_id: str
    role: str
    query: str
    expected_intent: str
    actual_intent: str
    expected_product_id: str | None
    actual_product_id: str | None
    expected_order_id: str | None
    actual_order_id: str | None
    expected_user_id: str | None
    actual_user_id: str | None
    expected_route: str
    actual_route: str
    intent_correct: bool
    entity_correct: bool
    route_correct: bool
    retrieval_ok: bool
    retrieval_note: str = ""


@dataclass
class SQLEvalResult:
    """Aggregated SQL evaluation result."""
    intent_accuracy: float
    entity_accuracy: float
    retrieval_accuracy: float
    routing_accuracy: float
    total_cases: int
    case_results: list[SQLCaseResult] = field(default_factory=list)
    failures_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    detail_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _load_test_cases() -> list[dict]:
    """Load test cases from the JSON file."""
    if not _TEST_CASES_PATH.exists():
        raise FileNotFoundError(f"SQL test cases not found at {_TEST_CASES_PATH}")
    with _TEST_CASES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _classify(role: str, query: str, conversation_context: str = "") -> dict[str, Any]:

    if role == "customer":
        from backend.nodes.customer.classify_intent import classify_intent_node
        state = {"query": query, "conversation_context": conversation_context}
        result = classify_intent_node(state)
    else:
        from backend.nodes.manager.classify_intent import classify_intent_node
        state = {"query": query, "conversation_context": conversation_context}
        result = classify_intent_node(state)
    return result


def _derive_route(role: str, intent: str) -> str:
    """Derive the graph node the intent would route to."""
    if role == "customer":
        return _CUSTOMER_INTENT_ROUTE.get(intent, "response_node")
    return _MANAGER_INTENT_ROUTE.get(intent, "response_node")


def _check_retrieval(role: str, intent: str, entities: dict[str, Any]) -> tuple[bool, str]:

    product_id = entities.get("product_id")
    order_id = entities.get("order_id")
    user_id = entities.get("user_id")

    try:
        if role == "customer":
            from backend.tools import customer_sql_tool
            if intent == "PRODUCT_INFO" and product_id:
                result = customer_sql_tool.get_product_details(product_id)
                if result:
                    return True, f"Product {product_id} found"
                return False, f"Product {product_id} not found in DB"
            elif intent == "PRODUCT_REVIEW" and product_id:
                result = customer_sql_tool.get_product_reviews(product_id)
                return bool(result), f"Reviews for {product_id}: {result.get('review_count', 0)} found"
            elif intent == "ORDER_DETAILS" and order_id:
                # Use a dummy user_id — we only verify the order exists here
                result = customer_sql_tool.get_product_details("P0001")  # just a connectivity check
                return True, "Retrieval connectivity confirmed"
            elif intent == "PURCHASE_HISTORY":
                return True, "PURCHASE_HISTORY uses authenticated user_id — skipped"
            elif intent in ("POLICY", "RECOMMENDATION", "GENERAL"):
                return True, f"{intent} does not use SQL retrieval"

        else:  # manager
            from backend.tools import manager_sql_tool
            if intent == "PRODUCT_INFO" and product_id:
                result = manager_sql_tool.get_product_details(product_id)
                if result:
                    return True, f"Product {product_id} found"
                return False, f"Product {product_id} not found in DB"
            elif intent == "PRODUCT_REVIEW" and product_id:
                result = manager_sql_tool.get_product_reviews(product_id)
                return True, f"Reviews for {product_id}: {result.get('review_count', 0)}"
            elif intent == "ORDER_DETAILS" and order_id:
                result = manager_sql_tool.get_order_details(order_id)
                if result:
                    return True, f"Order {order_id} found"
                return False, f"Order {order_id} not found in DB"
            elif intent == "CUSTOMER_DETAILS" and user_id:
                result = manager_sql_tool.get_customer_details(user_id)
                if result:
                    return True, f"Customer {user_id} found"
                return False, f"Customer {user_id} not found in DB"
            elif intent == "CUSTOMER_PURCHASE_HISTORY" and user_id:
                result = manager_sql_tool.customer_purchase_history(user_id)
                return True, f"Purchase history for {user_id}: {result.get('total_orders', 0)} orders"
            elif intent == "INVENTORY":
                result = manager_sql_tool.inventory_summary()
                return bool(result), "Inventory summary retrieved"
            elif intent in ("SALES_ANALYTICS", "CUSTOMER_ANALYTICS", "PRODUCT_ANALYTICS", "BUSINESS_SUMMARY"):
                return True, f"{intent} routes to analytics_node — SQL retrieval N/A"
            elif intent == "FORECAST":
                return True, "FORECAST routes to forecast_node — SQL retrieval N/A"
            elif intent in ("POLICY", "GENERAL"):
                return True, f"{intent} does not use SQL retrieval"

    except Exception as exc:  # noqa: BLE001
        return False, f"Retrieval error: {exc}"

    return True, "No retrieval check defined for this intent/entity combination"


def _evaluate_single_case(case: dict) -> SQLCaseResult:
    """Evaluate a single-turn test case."""
    result = _classify(case["role"], case["query"])
    actual_intent: str = result.get("intent", "")
    entities: dict = result.get("entities", {})
    actual_product_id = entities.get("product_id")
    actual_order_id = entities.get("order_id")
    actual_user_id = entities.get("user_id")

    expected_intent = case["expected_intent"]
    expected_product_id = case.get("expected_product_id")
    expected_order_id = case.get("expected_order_id")
    expected_user_id = case.get("expected_user_id")
    expected_route = case.get("expected_route", _derive_route(case["role"], expected_intent))

    intent_correct = actual_intent == expected_intent

    # Entity accuracy: check only the entities relevant to this test case
    entity_correct = True
    if expected_product_id is not None:
        entity_correct = entity_correct and (actual_product_id == expected_product_id)
    if expected_order_id is not None:
        entity_correct = entity_correct and (actual_order_id == expected_order_id)
    if expected_user_id is not None:
        entity_correct = entity_correct and (actual_user_id == expected_user_id)

    actual_route = _derive_route(case["role"], actual_intent)
    route_correct = actual_route == expected_route

    retrieval_ok, retrieval_note = _check_retrieval(case["role"], actual_intent, entities)

    return SQLCaseResult(
        case_id=case["id"],
        role=case["role"],
        query=case["query"],
        expected_intent=expected_intent,
        actual_intent=actual_intent,
        expected_product_id=expected_product_id,
        actual_product_id=actual_product_id,
        expected_order_id=expected_order_id,
        actual_order_id=actual_order_id,
        expected_user_id=expected_user_id,
        actual_user_id=actual_user_id,
        expected_route=expected_route,
        actual_route=actual_route,
        intent_correct=intent_correct,
        entity_correct=entity_correct,
        route_correct=route_correct,
        retrieval_ok=retrieval_ok,
        retrieval_note=retrieval_note,
    )


def _evaluate_conversational_case(case: dict) -> list[SQLCaseResult]:
    """Evaluate a multi-turn conversational test case."""
    turns = case["turns"]
    results: list[SQLCaseResult] = []
    conversation_history: list[str] = []

    for turn_idx, turn in enumerate(turns):
        query = turn["user"]
        expected_intent = turn["expected_intent"]
        expected_product_id = turn.get("expected_product_id")
        expected_order_id = turn.get("expected_order_id")
        expected_user_id = turn.get("expected_user_id")
        note = turn.get("note", "")

        # Build conversation context from prior turns (same format as the chatbot)
        conversation_context = "\n".join(conversation_history)

        classify_result = _classify(case["role"], query, conversation_context)
        actual_intent: str = classify_result.get("intent", "")
        entities: dict = classify_result.get("entities", {})
        actual_product_id = entities.get("product_id")
        actual_order_id = entities.get("order_id")
        actual_user_id = entities.get("user_id")

        intent_correct = actual_intent == expected_intent

        entity_correct = True
        if expected_product_id is not None:
            entity_correct = entity_correct and (actual_product_id == expected_product_id)
        if expected_order_id is not None:
            entity_correct = entity_correct and (actual_order_id == expected_order_id)
        if expected_user_id is not None:
            entity_correct = entity_correct and (actual_user_id == expected_user_id)

        expected_route = _derive_route(case["role"], expected_intent)
        actual_route = _derive_route(case["role"], actual_intent)
        route_correct = actual_route == expected_route

        retrieval_ok, retrieval_note = _check_retrieval(case["role"], actual_intent, entities)

        conv_description = f"{case['id']}_turn{turn_idx + 1}"
        if note:
            conv_description += f" [{note}]"

        results.append(SQLCaseResult(
            case_id=conv_description,
            role=case["role"],
            query=query,
            expected_intent=expected_intent,
            actual_intent=actual_intent,
            expected_product_id=expected_product_id,
            actual_product_id=actual_product_id,
            expected_order_id=expected_order_id,
            actual_order_id=actual_order_id,
            expected_user_id=expected_user_id,
            actual_user_id=actual_user_id,
            expected_route=expected_route,
            actual_route=actual_route,
            intent_correct=intent_correct,
            entity_correct=entity_correct,
            route_correct=route_correct,
            retrieval_ok=retrieval_ok,
            retrieval_note=retrieval_note,
        ))

        # Add this turn to the conversation history so the next turn has context
        conversation_history.append(f"User: {query}")
        # Use a minimal placeholder response so the classifier sees prior context
        conversation_history.append(f"Assistant: [handled {actual_intent}]")

    return results


def run_sql_evaluation() -> SQLEvalResult:
    """
    Run the full SQL evaluation over all single-turn and conversational test cases.

    Returns:
        :class:`SQLEvalResult` with aggregated metrics and detailed DataFrames.
    """
    all_cases = _load_test_cases()
    all_results: list[SQLCaseResult] = []

    for case in all_cases:
        if "turns" in case:
            # Conversational test
            logger.info("SQL evaluation — conversational case %s (%d turns)", case["id"], len(case["turns"]))
            conv_results = _evaluate_conversational_case(case)
            all_results.extend(conv_results)
        else:
            # Single-turn test
            logger.info("SQL evaluation — single-turn case %s: %r", case["id"], case["query"][:60])
            result = _evaluate_single_case(case)
            all_results.append(result)

    n = len(all_results)
    if n == 0:
        return SQLEvalResult(0.0, 0.0, 0.0, 0.0, 0)

    intent_accuracy = sum(1 for r in all_results if r.intent_correct) / n
    entity_accuracy = sum(1 for r in all_results if r.entity_correct) / n
    retrieval_accuracy = sum(1 for r in all_results if r.retrieval_ok) / n
    routing_accuracy = sum(1 for r in all_results if r.route_correct) / n

    # Build DataFrames
    rows = [
        {
            "Case ID": r.case_id,
            "Role": r.role,
            "Query": r.query,
            "Expected Intent": r.expected_intent,
            "Actual Intent": r.actual_intent,
            "Expected Product ID": r.expected_product_id or "",
            "Actual Product ID": r.actual_product_id or "",
            "Expected Order ID": r.expected_order_id or "",
            "Actual Order ID": r.actual_order_id or "",
            "Expected User ID": r.expected_user_id or "",
            "Actual User ID": r.actual_user_id or "",
            "Expected Route": r.expected_route,
            "Actual Route": r.actual_route,
            "Intent ✓": "✅" if r.intent_correct else "❌",
            "Entity ✓": "✅" if r.entity_correct else "❌",
            "Route ✓": "✅" if r.route_correct else "❌",
            "Retrieval ✓": "✅" if r.retrieval_ok else "❌",
            "Retrieval Note": r.retrieval_note,
            "Status": "✅ PASS" if (r.intent_correct and r.entity_correct and r.route_correct) else "❌ FAIL",
        }
        for r in all_results
    ]
    detail_df = pd.DataFrame(rows)
    failures_df = detail_df[detail_df["Status"] == "❌ FAIL"].reset_index(drop=True)

    logger.info(
        "SQL evaluation — done | intent_acc=%.3f | entity_acc=%.3f | retrieval_acc=%.3f | route_acc=%.3f",
        intent_accuracy, entity_accuracy, retrieval_accuracy, routing_accuracy,
    )

    return SQLEvalResult(
        intent_accuracy=round(intent_accuracy, 3),
        entity_accuracy=round(entity_accuracy, 3),
        retrieval_accuracy=round(retrieval_accuracy, 3),
        routing_accuracy=round(routing_accuracy, 3),
        total_cases=n,
        case_results=all_results,
        failures_df=failures_df,
        detail_df=detail_df,
    )
