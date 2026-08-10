"""
Recommendation Service
======================
Service layer for personalised product recommendations.

Current strategy (Phase 1)
--------------------------
1. Fetch active products in the customer's ``preferred_category``,
   sorted by ``total_sold`` descending.
2. Fill any remaining slots (up to *limit*) with globally top-selling
   products not already in the list.

This two-tier approach is intentionally decoupled behind a clean interface
so the entire algorithm can be swapped for a collaborative-filtering or
ML-based model in a future phase without any changes to the UI layer.

Future phases
-------------
- Collaborative filtering  (user × product purchase matrix)
- Content-based filtering  (purchase history, browsing patterns)
- LangGraph recommendation agent with tool calls
- scikit-learn / LightFM model integration
"""

from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.models import Order, OrderItem, Product
from backend.repositories.product_repository import ProductRepository
from backend.repositories.user_repository import UserRepository

logger = get_logger(__name__)

_product_repo = ProductRepository()
_user_repo = UserRepository()


def get_recommended_products(
    db: Session,
    user_id: str,
    limit: int = 6,
) -> dict:
    """
    Return *limit* personalised product recommendations for *user_id*.

    Algorithm (Phase 1 — category-based top-seller):

    1. Look up the customer's ``preferred_category``.
    2. Fetch active products in that category sorted by ``total_sold`` desc,
       excluding products the customer has already purchased.
    3. Fill any remaining slots with globally top-selling active products
       (excluding already-included products and previously purchased ones).

    Args:
        db:      Active SQLAlchemy session.
        user_id: Authenticated customer identifier.
        limit:   Maximum number of recommendations to return (default 6).

    Returns:
        Dict with:

        - ``user_id``              : str
        - ``preferred_category``   : str — the category used for personalisation
        - ``strategy``             : str — algorithm identifier
        - ``strategy_description`` : str — human-readable explanation
        - ``count``                : int
        - ``recommendations``      : list[dict] — product dicts
    """
    logger.info(
        "RecommendationService.get_recommended_products — user_id=%s limit=%d",
        user_id, limit,
    )

    user = _user_repo.get_by_id(db, user_id)
    preferred_cat = (user.preferred_category or "") if user else ""

    # Products the customer has already purchased — exclude from recs.
    purchased_ids: set[str] = _get_purchased_product_ids(db, user_id)

    recommendations: list[dict] = []

    # --- Tier 1: preferred category top-sellers ---
    if preferred_cat:
        tier1 = _get_top_by_category(
            db,
            category=preferred_cat,
            exclude_ids=purchased_ids,
            limit=limit,
        )
        recommendations.extend(tier1)
        logger.debug(
            "Tier-1 (category=%s): %d products", preferred_cat, len(tier1)
        )

    # --- Tier 2: global top-sellers (fill remaining slots) ---
    remaining = limit - len(recommendations)
    if remaining > 0:
        already_included = {r["product_id"] for r in recommendations}
        tier2 = _get_global_top_sellers(
            db,
            exclude_ids=purchased_ids | already_included,
            limit=remaining,
        )
        recommendations.extend(tier2)
        logger.debug("Tier-2 (global): %d products", len(tier2))

    strategy = (
        "category_top_selling"
        if preferred_cat and recommendations
        else "global_top_selling"
    )
    strategy_desc = (
        f"Products from your favourite category ({preferred_cat}), "
        "topped up with our globally best-selling items."
        if preferred_cat
        else "Our globally best-selling products — personalised recommendations coming soon."
    )

    return {
        "user_id":              user_id,
        "preferred_category":   preferred_cat,
        "strategy":             strategy,
        "strategy_description": strategy_desc,
        "count":                len(recommendations),
        "recommendations":      recommendations,
    }


def recommend_for_customer(
    db: Session,
    user_id: str,
    limit: int = 8,
) -> dict:
    """
    Backward-compatible alias used by the LangGraph customer SQL tool.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.
        limit:   Max products to return.

    Returns:
        Same contract as :func:`get_recommended_products`.
    """
    return get_recommended_products(db, user_id, limit=limit)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_purchased_product_ids(db: Session, user_id: str) -> set[str]:
    """Return the set of product_ids this customer has already ordered."""
    rows = (
        db.query(OrderItem.product_id)
        .join(Order, OrderItem.order_id == Order.order_id)
        .filter(Order.user_id == user_id)
        .distinct()
        .all()
    )
    return {r.product_id for r in rows}


def _get_top_by_category(
    db: Session,
    category: str,
    exclude_ids: set[str],
    limit: int,
) -> list[dict]:
    """Top-selling active products in *category*, excluding *exclude_ids*."""
    from sqlalchemy import func as sa_func

    rows = (
        db.query(
            Product.product_id,
            Product.product_name,
            Product.brand,
            Product.category,
            Product.sub_category,
            Product.final_price,
            Product.price,
            Product.discount_percentage,
            Product.rating,
            Product.total_reviews,
            Product.image_url,
            Product.total_sold,
        )
        .filter(
            Product.status == "Active",
            Product.category == category,
            Product.stock_quantity > 0,
        )
        .order_by(Product.total_sold.desc())
        .limit(limit + len(exclude_ids) + 10)   # over-fetch, then filter
        .all()
    )
    result = []
    for r in rows:
        if r.product_id in exclude_ids:
            continue
        result.append(_fmt_product(r))
        if len(result) >= limit:
            break
    return result


def _get_global_top_sellers(
    db: Session,
    exclude_ids: set[str],
    limit: int,
) -> list[dict]:
    """Store-wide top-selling active products, excluding *exclude_ids*."""
    rows = (
        db.query(
            Product.product_id,
            Product.product_name,
            Product.brand,
            Product.category,
            Product.sub_category,
            Product.final_price,
            Product.price,
            Product.discount_percentage,
            Product.rating,
            Product.total_reviews,
            Product.image_url,
            Product.total_sold,
        )
        .filter(
            Product.status == "Active",
            Product.stock_quantity > 0,
        )
        .order_by(Product.total_sold.desc())
        .limit(limit + len(exclude_ids) + 20)
        .all()
    )
    result = []
    for r in rows:
        if r.product_id in exclude_ids:
            continue
        result.append(_fmt_product(r))
        if len(result) >= limit:
            break
    return result


def _fmt_product(r) -> dict:
    """Serialise a Product ORM row to a plain dict."""
    return {
        "product_id":          r.product_id,
        "product_name":        r.product_name,
        "brand":               r.brand,
        "category":            r.category,
        "sub_category":        r.sub_category or "",
        "price":               float(r.price or 0),
        "final_price":         float(r.final_price or 0),
        "discount_percentage": float(r.discount_percentage or 0),
        "rating":              float(r.rating or 0),
        "total_reviews":       int(r.total_reviews or 0),
        "total_sold":          int(r.total_sold or 0),
        "image_url":           r.image_url or "",
    }
