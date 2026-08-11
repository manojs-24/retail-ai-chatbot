from __future__ import annotations

import logging
from typing import Any

from backend.guardrails.output_guard import run_output_guard

logger = logging.getLogger(__name__)


def output_guard_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Post-tool output guard LangGraph node.

    Reads:
        - ``state["response"]``            — raw LLM response
        - ``state["tool_result"]``         — raw JSON from SQL / tool node
        - ``state["retrieved_documents"]`` — docs from RAG node
        - ``state["intent"]``              — classified intent string
        - ``state["guard_blocked"]``       — skip if already blocked

    Writes:
        - ``state["response"]`` — sanitised / safe response string

    Args:
        state: Current LangGraph state dict.

    Returns:
        Partial state update dict with the cleaned ``"response"``.
    """
    # If a guard already blocked this request, the response is already a
    # clean, human-readable rejection message — no scrubbing needed.
    if state.get("guard_blocked", False):
        logger.debug("Output guard skipped — request was already blocked by an upstream guard.")
        return {}

    response: str = state.get("response", "")
    tool_result: str = state.get("tool_result", "")
    retrieved_documents: list[Any] = state.get("retrieved_documents", [])
    intent: str = state.get("intent", "")

    logger.info(
        "Output guard node — intent=%s response_len=%d", intent, len(response)
    )

    safe_response = run_output_guard(
        response=response,
        tool_result=tool_result,
        retrieved_documents=retrieved_documents,
        intent=intent,
    )

    return {"response": safe_response}
