from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Sensitive field name patterns
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

# "No results" fallback messages
_NO_SQL_RESULT_MSG = "No matching records found."
_NO_RAG_RESULT_MSG = (
    "I couldn't find relevant information in the company documents. "
    "Please contact our support team for assistance."
)
_GENERIC_EMPTY_MSG = "I'm sorry, I wasn't able to find any information for your request."


def scrub_sensitive_data(response: str) -> str:

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
    retrieved_documents = retrieved_documents or []
    rag_intents = {"POLICY"}
    sql_intents = {
        "PURCHASE_HISTORY", "ORDER_DETAILS", "PRODUCT_INFO",
        "PRODUCT_REVIEW", "RECOMMENDATION",
        "INVENTORY", "SALES_ANALYTICS", "CUSTOMER_ANALYTICS",
        "PRODUCT_ANALYTICS", "BUSINESS_SUMMARY", "FORECAST",
    }

    intent_upper = intent.upper()

    
    # 1. Empty SQL tool result                                             #
    
    if intent_upper in sql_intents and _is_empty_tool_result(tool_result):
        logger.info(
            "GUARDRAIL [output_guard] empty SQL result | intent=%s", intent_upper
        )
        return _NO_SQL_RESULT_MSG

    
    # 2. Empty RAG result                                                  #
    
    if intent_upper in rag_intents and _is_empty_rag_result(retrieved_documents):
        logger.info(
            "GUARDRAIL [output_guard] empty RAG result | intent=%s", intent_upper
        )
        return _NO_RAG_RESULT_MSG

    
    # 3. Sensitive data scrubbing                                          #
    
    safe_response = scrub_sensitive_data(response)

    logger.debug(
        "GUARDRAIL [output_guard] PASSED | intent=%s | response_len=%d",
        intent_upper, len(safe_response),
    )
    return safe_response
