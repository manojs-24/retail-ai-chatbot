"""
Injection Guard
===============
Detects two categories of injection attack in raw user input:

1. **Prompt injection** — attempts to override system instructions, leak the
   system prompt, or jailbreak the model.
2. **SQL injection** — attempts to embed raw SQL commands in user-facing text
   fields that could be forwarded to the database or an LLM-as-SQL system.

All checks are pattern-based (regex + keyword sets) — no LLM is called here.
Both functions return a ``(blocked: bool, reason: str)`` tuple so callers can
decide how to respond (log, short-circuit, etc.).

Design principles
-----------------
- Pure functions — no side effects other than logging.
- Case-insensitive matching.
- Easily extended: add patterns to the module-level sets/lists.
- Unit-test friendly: pass any string, inspect the tuple.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt-injection pattern registry
# ---------------------------------------------------------------------------
# Each entry is a compiled regex.  Patterns are anchored loosely so they
# match anywhere in the query string (not only at word boundaries).
_PROMPT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(previous|all|prior|above)\s+(instructions?|prompts?|rules?)", re.I),
    re.compile(r"forget\s+(your|all|previous|prior)\s+(instructions?|prompts?|rules?|context)", re.I),
    re.compile(r"(reveal|show|print|display|repeat|output)\s+(your\s+)?(system\s+)?(prompt|instructions?)", re.I),
    re.compile(r"system\s+prompt", re.I),
    re.compile(r"developer\s+(mode|prompt|instructions?)", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"pretend\s+(to\s+be|you\s+are|you're)\s+", re.I),
    re.compile(r"\bDAN\b"),                                    # "Do Anything Now" jailbreak
    re.compile(r"ignore\s+safety", re.I),
    re.compile(r"(act|behave)\s+as\s+(if\s+you\s+are|a)\s+", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"new\s+(role|persona|instructions?|identity)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|your)\s+", re.I),
    re.compile(r"override\s+(your\s+)?(instructions?|settings?|prompt)", re.I),
    re.compile(r"break\s+(out\s+of|free\s+from)\s+", re.I),
    re.compile(r"(token|secret|api[_\s]?key)\s*(leak|dump|reveal|show)", re.I),
    re.compile(r"what\s+(are\s+)?your\s+(instructions?|rules?|guidelines?|constraints?)", re.I),
    re.compile(r"end\s+(the\s+)?(conversation|session)\s+and\s+", re.I),
    re.compile(r"<\s*\|?\s*(system|endoftext|im_start)\s*\|?\s*>", re.I),  # token-stuffing
]

# ---------------------------------------------------------------------------
# SQL-injection keyword/pattern registry
# ---------------------------------------------------------------------------
_SQL_KEYWORDS: frozenset[str] = frozenset({
    "select", "drop", "delete", "update", "insert", "union",
    "truncate", "exec", "execute", "xp_", "sp_", "declare",
    "cast(", "convert(", "char(", "nchar(", "varchar(",
})

_SQL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"--\s*",                  re.I),   # SQL line comment
    re.compile(r"/\*.*?\*/",              re.I),   # SQL block comment
    re.compile(r";\s*(drop|delete|update|insert|select|truncate)", re.I),
    re.compile(r"\bor\s+1\s*=\s*1\b",    re.I),   # classic tautology
    re.compile(r"\bor\s+'\w+'\s*=\s*'\w+'\b", re.I),  # string tautology
    re.compile(r"\band\s+1\s*=\s*1\b",   re.I),
    re.compile(r"'\s*(or|and)\s*'",       re.I),   # quote-terminated injection
    re.compile(r"union\s+(all\s+)?select", re.I),
    re.compile(r"information_schema",     re.I),
    re.compile(r"sys\.(tables|columns|databases|objects)", re.I),
    re.compile(r"0x[0-9a-fA-F]{2,}",     re.I),   # hex-encoded payloads
]


def check_prompt_injection(query: str) -> tuple[bool, str]:
    """
    Scan *query* for prompt-injection patterns.

    Args:
        query: Raw user input string.

    Returns:
        ``(True, reason)`` if a prompt-injection attempt is detected,
        ``(False, "")``    if the query is clean.

    Example::

        blocked, reason = check_prompt_injection("ignore previous instructions")
        # blocked=True, reason="Prompt injection detected: ..."
    """
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(query):
            reason = f"Prompt injection detected: matched pattern '{pattern.pattern}'"
            logger.warning("GUARDRAIL [prompt_injection] | reason=%s | query=%r", reason, query[:120])
            return True, reason
    return False, ""


def check_sql_injection(query: str) -> tuple[bool, str]:
    """
    Scan *query* for SQL-injection patterns.

    Checks both keyword presence (token-level) and structural regex patterns
    (comment sequences, tautologies, UNION selects, etc.).

    Args:
        query: Raw user input string.

    Returns:
        ``(True, reason)`` if a SQL-injection attempt is detected,
        ``(False, "")``    if the query is clean.

    Example::

        blocked, reason = check_sql_injection("'; DROP TABLE users; --")
        # blocked=True, reason="SQL injection detected: ..."
    """
    lowered = query.lower()

    # Keyword check — look for isolated SQL keywords that have no business
    # appearing in a retail chatbot conversation.
    for kw in _SQL_KEYWORDS:
        # Use word-boundary matching for standalone keywords (e.g. "select")
        # but direct substring for multi-char tokens like "cast(" or "xp_".
        boundary_pattern = rf"\b{re.escape(kw)}\b" if kw.isalpha() else re.escape(kw)
        if re.search(boundary_pattern, lowered):
            reason = f"SQL injection detected: keyword '{kw}' found in query"
            logger.warning("GUARDRAIL [sql_injection] | reason=%s | query=%r", reason, query[:120])
            return True, reason

    # Structural pattern check
    for pattern in _SQL_PATTERNS:
        if pattern.search(query):
            reason = f"SQL injection detected: matched pattern '{pattern.pattern}'"
            logger.warning("GUARDRAIL [sql_injection] | reason=%s | query=%r", reason, query[:120])
            return True, reason

    return False, ""
