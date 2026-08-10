"""
Output Guard
============
Validates and sanitises the LLM response *before* it is surfaced to the user.

Responsibilities
----------------
1. **Sensitive data scrubbing** — remove or redact any passwords, API keys,
   tokens, database connection strings, or secret-like values that may have
   leaked into the LLM's output.

2. **Empty result handling** — return a structured "no results" message
   instead of an empty, hallucinated, or confusing LLM response when the
   underlying tool returned no data.

3. **Source-aware messaging** — distinguish between empty SQL results
   (``tool_result`` is ``"{}"`` or ``"[]"``) and empty RAG results
   (``retrieved_documents`` is empty) so the user gets an accurate explanation.

Design principles
-----------------
- Pure functions — no I/O other than logging.
- Regex-based scrubbing — deterministic, auditable, no LLM calls.
- Returns a plain ``str`` (the cleaned response) for easy drop-in use.
- Unit-test friendly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensitive field name patterns
# ---------------------------------------------------------------------------
# These patterns match field names that should NEVER appear in output.
# The regex replaces the value portion (after : or =) with [REDACTED].
_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    # JSON-style  "password": "abc123"  or  'password_hash': '...'
    re.compile(
        r'("|\')(?:password(?:_hash)?|api[_\s]?key|token|secret|db[_\s]?url'
        r'|database[_\s]?url|connection[_\s]?string|auth[_\s]?token'
        r'|access[_\s]?token|refresh[_\s]?token|private[_\s]?key'
        r'|client[_\s]?secret|jwt[_\s]?secret|signing[_\s]?key)\1\s*:\s*("|\')[^"\']{1,200}\2',
        re.IGNORECASE,
    ),
    # Bare word assignment   password = abc123
    re.compile(
        r'\b(?:password(?:_hash)?|api[_\s]?key|token|secret|db[_\s]?url'
        r'|database[_\s]?url|connection[_\s]?string|auth[_\s]?token'
        r'|access[_\s]?token|refresh[_\s]?token|private[_\s]?key'
        r'|client[_\s]?secret|jwt[_\s]?secret|signing[_\s]?key)\b\s*[=:]\s*\S+',
        re.IGNORECASE,
    ),
    # SQLite / PostgreSQL / MySQL connection strings
    re.compile(
        r'(?:sqlite|postgresql|mysql|mssql|oracle)\+?\w*://[^\s"\'<>]{3,200}',
        re.IGNORECASE,
    ),
    # Raw bearer / JWT tokens (3-part base64url)
    re.compile(
        r'\bBearer\s+[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+',
        re.IGNORECASE,
    ),
    # OpenAI-style API keys  sk-...
    re.compile(r'\bsk-[A-Za-z0-9]{20,}', re.IGNORECASE),
]

# Replacement token inserted wherever a sensitive value is scrubbed.
_REDACTED = "[REDACTED]"

# ---------------------------------------------------------------------------
# "No results" fallback messages
# ---------------------------------------------------------------------------
_NO_SQL_RESULT_MSG = "No matching records found."
_NO_RAG_RESULT_MSG = (
    "I couldn't find relevant information in the company documents. "
    "Please contact our support team for assistance."
)
_GENERIC_EMPTY_MSG = "I'm sorry, I wasn't able to find any information for your request."


def scrub_sensitive_data(response: str) -> str:
    """
    Scan *response* for sensitive field patterns and replace values with
    ``[REDACTED]``.

    Args:
        response: The raw LLM-generated response string.

    Returns:
        The sanitised response string.  If nothing sensitive is found,
        the original string is returned unchanged.

    Example::

        clean = scrub_sensitive_data('Your password is: "hunter2"')
        # Returns: 'Your password is: [REDACTED]'
    """
    cleaned = response
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(cleaned):
            cleaned = pattern.sub(_REDACTED, cleaned)
            logger.warning(
                "GUARDRAIL [output_guard] Sensitive data scrubbed | pattern=%s",
                pattern.pattern[:60],
            )
    return cleaned


def _is_empty_tool_result(tool_result: str) -> bool:
    """
    Return ``True`` if *tool_result* represents a meaningfully empty response.

    Handles:
    - Empty string ``""``
    - JSON empty object ``"{}"``
    - JSON empty array ``"[]"``
    - JSON object where all top-level list values are empty
    """
    if not tool_result or tool_result.strip() in ("{}", "[]", ""):
        return True
    try:
        parsed = json.loads(tool_result)
        if isinstance(parsed, dict):
            # Check if every list-valued key is empty
            list_keys = [v for v in parsed.values() if isinstance(v, list)]
            if list_keys and all(len(lst) == 0 for lst in list_keys):
                return True
        elif isinstance(parsed, list):
            return len(parsed) == 0
    except (json.JSONDecodeError, TypeError):
        pass
    return False


def _is_empty_rag_result(retrieved_documents: list[Any]) -> bool:
    """Return ``True`` if the RAG retriever returned no documents."""
    return not retrieved_documents


def run_output_guard(
    response: str,
    tool_result: str = "",
    retrieved_documents: list[Any] | None = None,
    intent: str = "",
) -> str:
    """
    Run the full output guard pipeline and return the final safe response.

    Steps:
    1. If ``tool_result`` is empty and intent is a SQL-type intent → return
       :data:`_NO_SQL_RESULT_MSG`.
    2. If ``retrieved_documents`` is empty and intent is ``"POLICY"`` →
       return :data:`_NO_RAG_RESULT_MSG`.
    3. Scrub any sensitive data patterns from *response*.
    4. Return the clean response.

    Args:
        response:             LLM-generated response string.
        tool_result:          Raw JSON string from the SQL / tool node.
        retrieved_documents:  Docs list from the RAG node.
        intent:               Classified intent string (for context-aware messaging).

    Returns:
        The sanitised, safe response string.

    Example::

        safe = run_output_guard(
            response="Here are your results...",
            tool_result="{}",
            intent="PURCHASE_HISTORY",
        )
        # Returns: "No matching records found."
    """
    retrieved_documents = retrieved_documents or []
    rag_intents = {"POLICY"}
    sql_intents = {
        "PURCHASE_HISTORY", "ORDER_DETAILS", "PRODUCT_INFO",
        "PRODUCT_REVIEW", "RECOMMENDATION",
        "INVENTORY", "SALES_ANALYTICS", "CUSTOMER_ANALYTICS",
        "PRODUCT_ANALYTICS", "BUSINESS_SUMMARY", "FORECAST",
    }

    intent_upper = intent.upper()

    # ------------------------------------------------------------------ #
    # 1. Empty SQL tool result                                             #
    # ------------------------------------------------------------------ #
    if intent_upper in sql_intents and _is_empty_tool_result(tool_result):
        logger.info(
            "GUARDRAIL [output_guard] empty SQL result | intent=%s", intent_upper
        )
        return _NO_SQL_RESULT_MSG

    # ------------------------------------------------------------------ #
    # 2. Empty RAG result                                                  #
    # ------------------------------------------------------------------ #
    if intent_upper in rag_intents and _is_empty_rag_result(retrieved_documents):
        logger.info(
            "GUARDRAIL [output_guard] empty RAG result | intent=%s", intent_upper
        )
        return _NO_RAG_RESULT_MSG

    # ------------------------------------------------------------------ #
    # 3. Sensitive data scrubbing                                          #
    # ------------------------------------------------------------------ #
    safe_response = scrub_sensitive_data(response)

    logger.debug(
        "GUARDRAIL [output_guard] PASSED | intent=%s | response_len=%d",
        intent_upper, len(safe_response),
    )
    return safe_response
