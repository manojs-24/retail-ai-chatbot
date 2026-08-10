"""
Customer graph state definition.

The :class:`CustomerState` TypedDict is the single shared data structure
that flows through every node in the customer LangGraph.  Each node reads
from and writes back into this dict — LangGraph merges the returned partial
dict onto the current state automatically.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class CustomerState(TypedDict, total=False):
    """
    Shared state for the customer chatbot graph.

    Fields
    ------
    query : str
        The raw natural-language question submitted by the customer.
    user_id : str
        Authenticated customer identifier (from ``st.session_state``).
    role : str
        Always ``"customer"`` — carried through for guard checks.
    intent : str
        Classified intent string (see :class:`~backend.schemas.customer_intent.CustomerIntent`).
        Set by the intent classifier node.
        Set to ``"BLOCKED"`` when a guard rejects the request.
    guard_blocked : bool
        ``True`` if any guardrail node has rejected this request.
        Downstream nodes check this flag to skip unnecessary work.
    entities : dict[str, str | None]
        Extracted entities from the classifier:

        - ``product_id``  — e.g. ``"P0023"`` or ``None``
        - ``order_id``    — e.g. ``"ORD00123"`` or ``None``
        - ``keyword``     — e.g. ``"laptop"`` or ``None``

        SQL nodes read entities from here instead of running regex on the raw query.
    retrieved_documents : list[dict[str, Any]]
        Serialised document chunks returned by the RAG retriever.
        Each dict contains ``page_content`` and ``metadata``.
    tool_result : str
        Raw output from whichever tool node handled the request
        (SQL, recommendation, analytics, …).  Empty string when unused.
    response : str
        Final natural-language answer delivered to the user.
        Written by the active tool node or the RAG node, then forwarded
        by the response node to the frontend.
    """

    query: str
    user_id: str
    role: str
    intent: str
    guard_blocked: bool
    entities: dict[str, Optional[str]]
    retrieved_documents: list[dict[str, Any]]
    tool_result: str
    response: str
    conversation_context: str
