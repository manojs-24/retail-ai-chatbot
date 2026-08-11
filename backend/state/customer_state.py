

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class CustomerState(TypedDict, total=False):

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
