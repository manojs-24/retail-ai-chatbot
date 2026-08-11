from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def response_node(state: dict[str, Any]) -> dict[str, Any]:

    response: str = state.get("response", "I'm sorry, I could not generate a response.")
    logger.debug("Response node — response length=%d", len(response))
    return {"response": response}
