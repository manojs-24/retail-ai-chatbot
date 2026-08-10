"""
Validation Guard
================
Validates and normalises entity identifiers extracted by the intent
classifier before they reach the SQL tool layer.

Supported formats
-----------------
+--------------+--------------------+----------------------+
| Entity       | DB Format          | Accepted inputs      |
+==============+====================+======================+
| product_id   | ``P0001``–``P0400``| ``P`` + 1–4 digits   |
| order_id     | ``ORD00001``–...   | ``ORD`` + 1–5 digits |
| user_id      | ``U0001``–``U0500``| ``U`` + 1–4 digits   |
+--------------+--------------------+----------------------+

All accepted inputs are zero-padded to the canonical DB width during
normalisation so the ORM query always gets the exact right format.

Design principles
-----------------
- Pure functions — no DB calls, no LLM calls.
- Returns structured ``EntityValidationResult`` dataclass for easy testing.
- Normalised IDs are returned even when the input had wrong padding, as long
  as the prefix and digit range are valid.
- Truly malformed inputs (wrong prefix, non-numeric suffix) are rejected.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns for each entity type
# ---------------------------------------------------------------------------
# Accept P + 1-to-N digits (normalisation pads to 4 digits).
_PRODUCT_RE = re.compile(r"^P(\d{1,6})$", re.IGNORECASE)
# Accept ORD + 1-to-N digits (normalisation pads to 5 digits).
_ORDER_RE = re.compile(r"^ORD(\d{1,8})$", re.IGNORECASE)
# Accept U + 1-to-N digits (normalisation pads to 4 digits).
_USER_RE = re.compile(r"^U(\d{1,6})$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EntityValidationResult:
    """
    The outcome of a :func:`validate_entities` call.

    Attributes
    ----------
    valid : bool
        ``True`` if all provided entity IDs are structurally valid.
    errors : list[str]
        Human-readable error messages for each rejected entity.
    normalised : dict[str, str | None]
        The corrected / zero-padded entity values.
        Keys are ``"product_id"``, ``"order_id"``, ``"user_id"``, ``"keyword"``.
        Absent or ``None`` values from the input are preserved as ``None``.
    """
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    normalised: dict[str, str | None] = field(default_factory=dict)


def _normalise_product_id(raw: str) -> str | None:
    """Return zero-padded product ID or ``None`` if format is invalid."""
    m = _PRODUCT_RE.match(raw.strip())
    if not m:
        return None
    digits = m.group(1).lstrip("0") or "0"
    return f"P{digits.zfill(4)}"


def _normalise_order_id(raw: str) -> str | None:
    """Return zero-padded order ID or ``None`` if format is invalid."""
    m = _ORDER_RE.match(raw.strip())
    if not m:
        return None
    digits = m.group(1).lstrip("0") or "0"
    return f"ORD{digits.zfill(5)}"


def _normalise_user_id(raw: str) -> str | None:
    """Return zero-padded user ID or ``None`` if format is invalid."""
    m = _USER_RE.match(raw.strip())
    if not m:
        return None
    digits = m.group(1).lstrip("0") or "0"
    return f"U{digits.zfill(4)}"


def validate_entities(
    entities: dict[str, str | None],
) -> EntityValidationResult:
    """
    Validate and normalise entity IDs extracted from the classifier.

    Iterates over the ``entities`` dict and validates each recognised key.
    Unknown keys are passed through unchanged.  ``None`` values are always
    passed through (they indicate the entity was not mentioned by the user).

    Args:
        entities: Dict produced by the intent classifier, e.g.::

            {
                "product_id": "P00023",
                "order_id":   None,
                "user_id":    None,
                "keyword":    "laptop",
            }

    Returns:
        :class:`EntityValidationResult` with ``valid``, ``errors``, and
        ``normalised`` fields.

    Example::

        result = validate_entities({"product_id": "PXYZ", "order_id": None})
        # result.valid=False
        # result.errors=["product_id 'PXYZ' is invalid ..."]
        # result.normalised={"product_id": None, "order_id": None}
    """
    result = EntityValidationResult()
    normalised: dict[str, str | None] = {}

    validators: dict[str, tuple[str, callable]] = {
        "product_id": ("P0001–P9999 (e.g. P0023)",  _normalise_product_id),
        "order_id":   ("ORD00001–ORD99999 (e.g. ORD00123)", _normalise_order_id),
        "user_id":    ("U0001–U9999 (e.g. U0001)",  _normalise_user_id),
    }

    for key, value in entities.items():
        if value is None or key not in validators:
            # Pass through None values and unrecognised keys (e.g. "keyword")
            normalised[key] = value
            continue

        fmt_hint, normalise_fn = validators[key]
        normalised_value = normalise_fn(value)

        if normalised_value is None:
            error = (
                f"'{key}' value '{value}' is not a valid ID. "
                f"Expected format: {fmt_hint}."
            )
            result.errors.append(error)
            result.valid = False
            normalised[key] = None
            logger.warning(
                "GUARDRAIL [validation_guard] | rejected %s=%r | %s", key, value, error
            )
        else:
            normalised[key] = normalised_value

    result.normalised = normalised
    return result
