"""
backend.guardrails — Defence-in-depth layer for the Retail AI chatbot pipeline.

All guardrails are pure functions with no side effects beyond logging.
They are composed by the input/output guard orchestrators and wired into
the LangGraph pipeline as dedicated nodes that run before and after every
tool execution.

Sub-modules
-----------
injection_guard
    Detects prompt-injection and SQL-injection patterns.
role_guard
    Enforces role-based intent permissions (customer vs manager).
validation_guard
    Validates and normalises entity IDs (product_id, order_id, user_id).
input_guard
    Orchestrates all pre-LLM checks: length, injection, topic, role, IDs.
output_guard
    Scrubs sensitive data from LLM responses and handles empty results.
"""

from backend.guardrails.injection_guard import check_prompt_injection, check_sql_injection
from backend.guardrails.role_guard import is_customer_allowed, is_manager_allowed
from backend.guardrails.validation_guard import validate_entities
from backend.guardrails.input_guard import run_input_guard, GuardResult
from backend.guardrails.output_guard import run_output_guard

__all__ = [
    "check_prompt_injection",
    "check_sql_injection",
    "is_customer_allowed",
    "is_manager_allowed",
    "validate_entities",
    "run_input_guard",
    "run_output_guard",
    "GuardResult",
]
