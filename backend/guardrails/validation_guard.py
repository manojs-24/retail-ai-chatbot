

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Compiled patterns for each entity type
# Accept P + 1-to-N digits (normalisation pads to 4 digits).
_PRODUCT_RE = re.compile(r"^P(\d{1,6})$", re.IGNORECASE)
# Accept ORD + 1-to-N digits (normalisation pads to 5 digits).
_ORDER_RE = re.compile(r"^ORD(\d{1,8})$", re.IGNORECASE)
# Accept U + 1-to-N digits (normalisation pads to 4 digits).
_USER_RE = re.compile(r"^U(\d{1,6})$", re.IGNORECASE)


# Result dataclass

@dataclass
class EntityValidationResult:
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
