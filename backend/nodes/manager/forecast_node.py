"""
Manager Forecast Node
=====================
LangGraph node for the manager chatbot — FORECAST intent.

Uses the existing :mod:`backend.ml.sales_forecast` LinearRegression model
to predict next-30-day revenue from historical monthly sales data.

Flow
----
    FORECAST intent
        → fetch monthly_sales from OrderRepository (via AnalyticsService)
        → pass to ml.sales_forecast.run()
        → enrich result with historical summary
        → narrate via gpt-4o-mini

Validation
----------
If fewer than 3 months of historical data are available, the node returns
a clear "insufficient data" message without attempting to fit the model.

Debug logging
-------------
Every invocation logs:
    - intent
    - historical_rows count
    - model used
    - forecast_30d value
    - execution time
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.ml import sales_forecast
from backend.tools import manager_sql_tool

load_dotenv()

logger = logging.getLogger(__name__)

_MIN_MONTHS_REQUIRED = 3

# ---------------------------------------------------------------------------
# System prompt — LLM explains the forecast clearly and honestly
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an intelligent retail analytics assistant for store managers at RetailHub Technologies.

You will be given a structured sales forecast result computed by a Linear Regression model
trained on historical monthly revenue data.

Your job:
1. Explain what the forecast predicts in clear, professional language.
2. Format the predicted revenue in Indian Rupees (₹), using L (lakh) or Cr (crore) for large values.
3. Mention the trend direction (up/down/stable) and what it means.
4. ALWAYS label the prediction as an estimate or projection — never present it as certain.
5. Show the last few months of actual revenue as context.
6. Be concise, honest, and professional."""


def _narrate(tool_data: dict, query: str) -> str:
    """Call the LLM to explain the forecast result."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
    data_str = json.dumps(tool_data, indent=2, default=str)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Manager question: {query}\n\n"
                f"Forecast result:\n{data_str}"
            )
        ),
    ]
    result = llm.invoke(messages)
    return result.content  # type: ignore[return-value]


def forecast_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Manager forecast node — real implementation.

    Fetches historical monthly sales from the database, runs the
    LinearRegression forecast model, and returns a narrated prediction.

    Args:
        state: Current LangGraph state.  Must contain ``"query"``.

    Returns:
        Partial state dict with ``"tool_result"`` (raw JSON) and
        ``"response"`` (LLM-narrated forecast explanation).
    """
    t_start = time.perf_counter()
    query: str = state.get("query", "")
    intent: str = state.get("intent", "FORECAST")

    logger.info(
        "Forecast node START | intent=%s | query=%r", intent, query[:80]
    )

    # ------------------------------------------------------------------
    # Step 1 — Fetch historical monthly sales
    # ------------------------------------------------------------------
    logger.info("Forecast node → fetching monthly_sales from DB")
    monthly_result = manager_sql_tool.monthly_sales()
    monthly_data: list[dict] = monthly_result.get("months", [])
    historical_rows = len(monthly_data)

    logger.info(
        "Forecast node — historical_rows=%d", historical_rows
    )

    # ------------------------------------------------------------------
    # Step 2 — Validate: need at least 3 months
    # ------------------------------------------------------------------
    if historical_rows < _MIN_MONTHS_REQUIRED:
        msg = (
            f"Not enough historical sales data to generate a reliable forecast. "
            f"Found {historical_rows} month(s) of data; at least {_MIN_MONTHS_REQUIRED} required."
        )
        logger.warning("Forecast node — insufficient data: %d months", historical_rows)
        return {"tool_result": "{}", "response": msg}

    # ------------------------------------------------------------------
    # Step 3 — Run the ML forecast model
    # ------------------------------------------------------------------
    logger.info(
        "Forecast node → running sales_forecast.run() on %d months", historical_rows
    )
    forecast_result = sales_forecast.run(monthly_data)

    model_used = forecast_result.get("model", "LinearRegression")
    forecast_30d = forecast_result.get("forecast_30d", 0.0)
    forecast_fmt = forecast_result.get("forecast_30d_fmt", "₹0")

    logger.info(
        "Forecast node RESULT | model=%s | forecast_30d=%.2f (%s) | trend=%s | elapsed=%.0fms",
        model_used,
        forecast_30d,
        forecast_fmt,
        forecast_result.get("trend", "stable"),
        (time.perf_counter() - t_start) * 1000,
    )

    # Enrich the result with metadata for the LLM
    tool_data = {
        **forecast_result,
        "historical_months_used": historical_rows,
        "forecast_horizon_days": 30,
        "note": (
            "This is a statistical estimate based on historical trends. "
            "Actual results may vary due to market conditions, seasonal factors, "
            "and business decisions."
        ),
    }

    # ------------------------------------------------------------------
    # Step 4 — LLM narration
    # ------------------------------------------------------------------
    response = _narrate(tool_data, query)
    tool_result = json.dumps(tool_data, default=str)

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Forecast node DONE | response_len=%d | total=%.0fms",
        len(response), total_ms,
    )

    return {"tool_result": tool_result, "response": response}
