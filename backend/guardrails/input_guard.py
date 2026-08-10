"""
Input Guard
===========
Orchestrates all pre-LLM security and validation checks.

Check pipeline (executed in order — first failure short-circuits):
    1. Empty query
    2. Query length  (max 2 000 characters)
    3. Prompt injection  (via :mod:`injection_guard`)
    4. SQL injection     (via :mod:`injection_guard`)
    5. Unsupported topic (off-topic retail check)
    6. Role-based intent check  (via :mod:`role_guard`)
    7. Entity ID format validation (via :mod:`validation_guard`)

The public API is :func:`run_input_guard`, which returns a :class:`GuardResult`.
All individual checkers are also importable for unit testing.

Logging
-------
Every rejection is logged at WARNING level with:
    - timestamp (via logging framework)
    - role
    - reason
    - truncated query (first 120 chars)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from backend.guardrails.injection_guard import check_prompt_injection, check_sql_injection
from backend.guardrails.role_guard import is_customer_allowed, is_manager_allowed
from backend.guardrails.validation_guard import validate_entities, EntityValidationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH: int = 2_000

# ---------------------------------------------------------------------------
# Unsupported topic detection
# ---------------------------------------------------------------------------
# Keywords that indicate an off-topic query (politics, sports, etc.).
# Each entry is a lower-cased substring; OR logic — any match → block.
_OFF_TOPIC_KEYWORDS: frozenset[str] = frozenset({
    # Politics
    "election", "government", "president", "prime minister", "parliament",
    "congress", "senate", "democrat", "republican", "political party",
    "politics", "politician", "vote", "referendum", "constitution",
    # Sports
    "cricket", "football", "soccer", "basketball", "tennis", "baseball",
    "rugby", "golf", "swimming", "olympics", "world cup", "fifa",
    "ipl", "nba", "nfl", "premier league", "championship",
    # Medical / health
    "diagnosis", "symptoms", "treatment", "prescription", "medicine",
    "hospital", "doctor", "surgery", "disease", "cancer", "diabetes",
    "vaccine", "medication", "therapy", "psychiatry",
    # General knowledge / science
    "photosynthesis", "periodic table", "quantum physics", "evolution",
    "big bang", "gravitational", "theorem", "calculus",
    # Programming
    "python code", "write a function", "code snippet", "programming",
    "algorithm", "data structure", "machine learning model",
    "neural network architecture", "train a model", "write code",
    "debug my code", "compile", "runtime error",
    # Finance (non-retail)
    "stock market", "cryptocurrency", "bitcoin", "forex", "trading",
    "share price", "nifty", "sensex", "mutual fund",
    # Entertainment
    "movie", "song lyrics", "celebrity", "actor", "actress",
    "music band", "netflix", "anime", "manga",
})

# Keywords that anchor a query as clearly retail-related.
# If any is present the topic check passes immediately.
_RETAIL_ANCHOR_KEYWORDS: frozenset[str] = frozenset({
    "product", "order", "purchase", "cart", "checkout", "delivery",
    "shipping", "return", "refund", "warranty", "stock", "inventory",
    "invoice", "receipt", "discount", "offer", "price", "review",
    "rating", "recommend", "catalogue", "customer", "account",
    "analytics", "revenue", "sales", "forecast", "policy",
    "store", "shop", "retail", "supplier", "brand", "category",
})


def check_empty(query: str) -> tuple[bool, str]:
    """
    Reject blank or whitespace-only queries.

    Returns:
        ``(True, reason)`` if the query is empty, ``(False, "")`` otherwise.
    """
    if not query or not query.strip():
        return True, "Query is empty. Please type a question."
    return False, ""


def check_length(query: str) -> tuple[bool, str]:
    """
    Reject queries exceeding :data:`MAX_QUERY_LENGTH` characters.

    Returns:
        ``(True, reason)`` if too long, ``(False, "")`` otherwise.
    """
    if len(query) > MAX_QUERY_LENGTH:
        reason = (
            f"Query is too long ({len(query)} characters). "
            f"Please keep questions under {MAX_QUERY_LENGTH} characters."
        )
        return True, reason
    return False, ""


def check_unsupported_topic(query: str) -> tuple[bool, str]:
    """
    Reject queries about topics unrelated to retail operations.

    Uses a two-pass approach:
    1. If any retail anchor keyword is present → allow immediately.
    2. If any off-topic keyword is present → reject.

    Args:
        query: Raw user input string.

    Returns:
        ``(True, reason)`` if the topic is off-topic, ``(False, "")`` otherwise.
    """
    lowered = query.lower()

    # Pass 1: retail anchor — if the query mentions retail concepts, let it through.
    for anchor in _RETAIL_ANCHOR_KEYWORDS:
        if anchor in lowered:
            return False, ""

    # Pass 2: off-topic keyword — block if a clearly off-topic term is found.
    for kw in _OFF_TOPIC_KEYWORDS:
        if kw in lowered:
            reason = (
                f"This assistant only handles retail-related questions "
                f"(products, orders, policies, recommendations, analytics, inventory). "
                f"Your question appears to be about '{kw}', which is outside our scope."
            )
            logger.warning(
                "GUARDRAIL [topic_check] | off_topic_kw=%r | query=%r", kw, query[:120]
            )
            return True, reason

    return False, ""


# ---------------------------------------------------------------------------
# GuardResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    """
    The outcome of :func:`run_input_guard`.

    Attributes
    ----------
    allowed : bool
        ``True`` if the query passed all checks and should proceed.
    rejection_reason : str
        Human-readable explanation if ``allowed`` is ``False``.
        Empty string when ``allowed`` is ``True``.
    normalised_entities : dict[str, str | None]
        The validated and zero-padded entity IDs.  Populated even when
        ``allowed`` is ``True`` so the SQL nodes receive clean values.
    guard_checks : list[str]
        Ordered list of check names that were executed (useful for debugging
        and audit logging).
    """
    allowed: bool = True
    rejection_reason: str = ""
    normalised_entities: dict[str, str | None] = field(default_factory=dict)
    guard_checks: list[str] = field(default_factory=list)


def run_input_guard(
    query: str,
    role: str,
    intent: str = "",
    entities: dict[str, Any] | None = None,
) -> GuardResult:
    """
    Run the full input guard pipeline against *query*.

    Checks are executed in order.  The first failure short-circuits — no
    further checks run.  If all checks pass, entity IDs are validated and
    normalised before being returned.

    Args:
        query:    Raw user input string.
        role:     Authenticated role — ``"customer"`` or ``"manager"``.
        intent:   The classified intent string (e.g. ``"POLICY"``).
                  Pass an empty string if intent is not yet known (pre-classification
                  checks will still run).
        entities: Dict of extracted entities from the classifier.
                  Defaults to an empty dict.

    Returns:
        :class:`GuardResult` — check ``allowed`` before proceeding.

    Example::

        result = run_input_guard(
            query="Show my orders",
            role="customer",
            intent="PURCHASE_HISTORY",
            entities={},
        )
        if not result.allowed:
            return {"response": result.rejection_reason}
    """
    entities = entities or {}
    guard = GuardResult(normalised_entities=dict(entities))

    # ------------------------------------------------------------------ #
    # 1. Empty query                                                       #
    # ------------------------------------------------------------------ #
    guard.guard_checks.append("empty_check")
    blocked, reason = check_empty(query)
    if blocked:
        _reject(guard, reason, role, query, "empty_check")
        return guard

    # ------------------------------------------------------------------ #
    # 2. Query length                                                      #
    # ------------------------------------------------------------------ #
    guard.guard_checks.append("length_check")
    blocked, reason = check_length(query)
    if blocked:
        _reject(guard, reason, role, query, "length_check")
        return guard

    # ------------------------------------------------------------------ #
    # 3. Prompt injection                                                  #
    # ------------------------------------------------------------------ #
    guard.guard_checks.append("prompt_injection_check")
    blocked, reason = check_prompt_injection(query)
    if blocked:
        _reject(guard, reason, role, query, "prompt_injection_check")
        return guard

    # ------------------------------------------------------------------ #
    # 4. SQL injection                                                     #
    # ------------------------------------------------------------------ #
    guard.guard_checks.append("sql_injection_check")
    blocked, reason = check_sql_injection(query)
    if blocked:
        _reject(guard, reason, role, query, "sql_injection_check")
        return guard

    # ------------------------------------------------------------------ #
    # 5. Unsupported topic                                                 #
    # ------------------------------------------------------------------ #
    guard.guard_checks.append("topic_check")
    blocked, reason = check_unsupported_topic(query)
    if blocked:
        _reject(guard, reason, role, query, "topic_check")
        return guard

    # ------------------------------------------------------------------ #
    # 6. Role-based intent check (only when intent is already known)      #
    # ------------------------------------------------------------------ #
    if intent:
        guard.guard_checks.append("role_check")
        if role == "customer":
            allowed_flag, reason = is_customer_allowed(intent)
        else:
            allowed_flag, reason = is_manager_allowed(intent)

        if not allowed_flag:
            _reject(guard, reason, role, query, "role_check")
            return guard

    # ------------------------------------------------------------------ #
    # 7. Entity ID validation                                              #
    # ------------------------------------------------------------------ #
    if entities:
        guard.guard_checks.append("entity_validation")
        val_result: EntityValidationResult = validate_entities(entities)
        guard.normalised_entities = val_result.normalised
        if not val_result.valid:
            reason = " | ".join(val_result.errors)
            _reject(guard, reason, role, query, "entity_validation")
            return guard

    logger.debug(
        "GUARDRAIL [input_guard] PASSED | role=%s | intent=%s | checks=%s",
        role, intent, guard.guard_checks,
    )
    return guard


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _reject(
    guard: GuardResult,
    reason: str,
    role: str,
    query: str,
    check_name: str,
) -> None:
    """Mark *guard* as rejected and emit a structured warning log."""
    guard.allowed = False
    guard.rejection_reason = reason
    logger.warning(
        "GUARDRAIL [input_guard] BLOCKED | check=%s | role=%s | reason=%s | query=%r",
        check_name, role, reason, query[:120],
    )
