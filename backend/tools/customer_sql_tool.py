"""
Customer SQL Tool
=================
Thin orchestration layer between the LangGraph customer SQL node and the
service layer.  Every function in this module:

1. Opens a scoped SQLAlchemy session.
2. Delegates to the appropriate service.
3. Closes the session in a ``finally`` block.
4. Returns a plain Python dict (JSON-serialisable).

Security guarantee
------------------
*user_id* is always taken from the LangGraph state (the authenticated
session), never from the user's text input.  No function in this module
accepts a user-supplied user_id.

Functions
---------
- get_purchase_history
- get_recent_orders
- get_order_details
- search_products
- get_product_details
- get_product_reviews
- recommend_products
"""

from __future__ import annotations

from backend.core.database import SessionLocal
from backend.core.logging import get_logger
from backend.services.order_service import OrderService
from backend.services.product_service import ProductService
from backend.services.recommendation_service import recommend_for_customer
from backend.services.user_service import UserService

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level service singletons (stateless — safe to reuse)
# ---------------------------------------------------------------------------
_order_svc = OrderService()
_product_svc = ProductService()
_user_svc = UserService()


def get_purchase_history(user_id: str) -> dict:
    """
    Return all orders for *user_id*, sorted newest first.

    Args:
        user_id: Authenticated customer identifier from LangGraph state.

    Returns:
        Dict with ``user_id``, ``total_orders``, and ``orders`` list.
    """
    logger.info("customer_sql_tool.get_purchase_history — user_id=%s", user_id)
    db = SessionLocal()
    try:
        return _order_svc.get_purchase_history(db, user_id)
    finally:
        db.close()


def get_recent_orders(user_id: str, limit: int = 5) -> dict:
    """
    Return the *limit* most recent orders for *user_id*.

    Args:
        user_id: Authenticated customer identifier from LangGraph state.
        limit:   Number of orders to return (default 5).

    Returns:
        Dict with ``user_id``, ``count``, and ``orders`` list.
    """
    logger.info(
        "customer_sql_tool.get_recent_orders — user_id=%s limit=%d",
        user_id, limit,
    )
    db = SessionLocal()
    try:
        return _order_svc.get_recent_orders(db, user_id, limit=limit)
    finally:
        db.close()


def get_order_details(order_id: str, user_id: str) -> dict | None:
    """
    Return full details for *order_id*, only if it belongs to *user_id*.

    Ownership is enforced by :class:`~backend.services.order_service.OrderService`.

    Args:
        order_id: Order identifier (may come from user input).
        user_id:  Authenticated customer identifier from LangGraph state.
                  NEVER sourced from user input.

    Returns:
        Dict with order fields and ``items`` list, or ``None`` if the
        order does not exist or does not belong to *user_id*.
    """
    logger.info(
        "customer_sql_tool.get_order_details — order_id=%s user_id=%s",
        order_id, user_id,
    )
    db = SessionLocal()
    try:
        return _order_svc.get_order_details(db, order_id, user_id)
    finally:
        db.close()


def search_products(keyword: str) -> dict:
    """
    Search the product catalogue by keyword.

    Searches across product name, brand, category, and sub_category.

    Args:
        keyword: Search term.

    Returns:
        Dict with ``keyword``, ``count``, and ``products`` list.
    """
    logger.info("customer_sql_tool.search_products — keyword=%r", keyword)
    db = SessionLocal()
    try:
        return _product_svc.search_products(db, keyword)
    finally:
        db.close()


def get_product_details(product_id: str) -> dict | None:
    """
    Return full product details with average rating and review count.

    Args:
        product_id: Product identifier.

    Returns:
        Dict with product fields plus ``avg_rating`` and ``review_count``,
        or ``None`` if the product does not exist.
    """
    logger.info(
        "customer_sql_tool.get_product_details — product_id=%s", product_id
    )
    db = SessionLocal()
    try:
        return _product_svc.get_product_details(db, product_id)
    finally:
        db.close()


def get_product_reviews(product_id: str, limit: int = 10) -> dict:
    """
    Return recent customer reviews for *product_id*.

    Args:
        product_id: Product identifier (e.g. ``"P0023"``).
        limit:      Maximum number of reviews to return (default 10).

    Returns:
        Dict with ``product_id``, ``avg_rating``, ``review_count``,
        ``count``, and ``reviews`` list.
    """
    logger.info(
        "customer_sql_tool.get_product_reviews — product_id=%s limit=%d",
        product_id, limit,
    )
    db = SessionLocal()
    try:
        return _product_svc.get_product_reviews(db, product_id, limit=limit)
    finally:
        db.close()


def recommend_products(user_id: str, limit: int = 8) -> dict:
    """
    Return product recommendations for *user_id*.

    Phase 1 returns top-selling products as a baseline recommendation.
    Personalised recommendations will be added in a future phase.

    Args:
        user_id: Authenticated customer identifier from LangGraph state.
        limit:   Number of products to recommend.

    Returns:
        Dict with ``user_id``, ``strategy``, ``strategy_description``,
        ``count``, and ``recommendations`` list.
    """
    logger.info(
        "customer_sql_tool.recommend_products — user_id=%s limit=%d",
        user_id, limit,
    )
    db = SessionLocal()
    try:
        return recommend_for_customer(db, user_id, limit=limit)
    finally:
        db.close()
