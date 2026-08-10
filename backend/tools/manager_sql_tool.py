"""
Manager SQL Tool
================
Thin orchestration layer between the LangGraph manager SQL node and the
service layer.  Every function in this module:

1. Opens a scoped SQLAlchemy session.
2. Delegates to the appropriate service.
3. Closes the session in a ``finally`` block.
4. Returns a plain Python dict (JSON-serialisable).

Access control
--------------
Managers may retrieve any customer's data.  There are no ownership
restrictions here — those belong in customer-facing tools only.

Sensitive data rule
-------------------
NEVER expose passwords, password_hash, api_key, or tokens.
UserService.get_customer_profile() enforces this — it never serialises
the password column.

Functions
---------
- inventory_summary
- low_stock_products
- top_selling_products
- category_summary
- sales_summary
- monthly_sales
- full_sales_analytics
- customer_analytics
- get_order_details
- get_product_details
- get_product_reviews
- get_customer_details
- customer_purchase_history
"""

from __future__ import annotations

import logging

from backend.core.database import SessionLocal
from backend.core.logging import get_logger
from backend.services.analytics_service import AnalyticsService
from backend.services.order_service import OrderService
from backend.services.product_service import ProductService
from backend.services.user_service import UserService

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level service singletons (stateless — safe to reuse)
# ---------------------------------------------------------------------------
_analytics_svc = AnalyticsService()
_order_svc = OrderService()
_product_svc = ProductService()
_user_svc = UserService()


def inventory_summary() -> dict:
    """Return a high-level inventory health snapshot."""
    logger.info("manager_sql_tool.inventory_summary")
    db = SessionLocal()
    try:
        return _analytics_svc.get_inventory_summary(db)
    finally:
        db.close()


def low_stock_products() -> dict:
    """Return all low-stock and out-of-stock products."""
    logger.info("manager_sql_tool.low_stock_products")
    db = SessionLocal()
    try:
        return _analytics_svc.get_low_stock_products(db)
    finally:
        db.close()


def top_selling_products(limit: int = 10) -> dict:
    """Return the top-selling products by units sold."""
    logger.info("manager_sql_tool.top_selling_products — limit=%d", limit)
    db = SessionLocal()
    try:
        return _analytics_svc.get_top_selling_products(db, limit=limit)
    finally:
        db.close()


def category_summary() -> dict:
    """Return product and stock counts grouped by category."""
    logger.info("manager_sql_tool.category_summary")
    db = SessionLocal()
    try:
        return _analytics_svc.get_category_summary(db)
    finally:
        db.close()


def sales_summary() -> dict:
    """Return store-wide sales KPIs: total revenue, total orders, avg order value."""
    logger.info("manager_sql_tool.sales_summary")
    db = SessionLocal()
    try:
        return _analytics_svc.get_sales_summary(db)
    finally:
        db.close()


def monthly_sales() -> dict:
    """Return sales grouped by calendar year-month, newest first."""
    logger.info("manager_sql_tool.monthly_sales")
    db = SessionLocal()
    try:
        return _analytics_svc.get_monthly_sales(db)
    finally:
        db.close()


def full_sales_analytics() -> dict:
    """
    Return a comprehensive sales analytics package.

    Includes: KPIs, monthly breakdown, top-selling products (by units and
    revenue), revenue by category, and trend direction.
    All calculations are done by SQLAlchemy + Pandas — NOT by the LLM.

    Returns:
        Dict with ``total_revenue``, ``total_orders``, ``avg_order_value``,
        ``monthly_breakdown``, ``top_selling_products``,
        ``top_revenue_products``, ``revenue_by_category``, ``sales_trend``.
    """
    logger.info("manager_sql_tool.full_sales_analytics")
    db = SessionLocal()
    try:
        return _analytics_svc.get_full_sales_analytics(db)
    finally:
        db.close()


def customer_analytics() -> dict:
    """
    Return customer analytics: highest-spending customers and customer count.

    Returns:
        Dict with ``total_customers`` and ``highest_spenders`` list.
    """
    logger.info("manager_sql_tool.customer_analytics")
    db = SessionLocal()
    try:
        return _analytics_svc.get_customer_analytics(db)
    finally:
        db.close()


def get_order_details(order_id: str) -> dict | None:
    """
    Return full details for any order — no ownership check (manager-only).

    Args:
        order_id: Order identifier (e.g. ``"ORD00123"``).

    Returns:
        Dict with order header, customer_id, items list, or ``None`` if
        the order does not exist.
    """
    logger.info("manager_sql_tool.get_order_details — order_id=%s", order_id)
    db = SessionLocal()
    try:
        return _order_svc.get_order_details_for_manager(db, order_id)
    finally:
        db.close()


def get_product_details(product_id: str) -> dict | None:
    """Return full product details with aggregate review statistics."""
    logger.info("manager_sql_tool.get_product_details — product_id=%s", product_id)
    db = SessionLocal()
    try:
        return _product_svc.get_product_details(db, product_id)
    finally:
        db.close()


def get_product_reviews(product_id: str, limit: int = 10) -> dict:
    """Return recent customer reviews for a product."""
    logger.info(
        "manager_sql_tool.get_product_reviews — product_id=%s limit=%d",
        product_id, limit,
    )
    db = SessionLocal()
    try:
        return _product_svc.get_product_reviews(db, product_id, limit=limit)
    finally:
        db.close()


def get_customer_details(user_id: str) -> dict | None:
    """
    Return a customer's safe profile — no passwords or secrets exposed.

    Args:
        user_id: Any customer's identifier.

    Returns:
        Dict with safe profile fields (user_id, full_name, email, phone,
        city, state, gender, age, join_date, total_orders, total_spent,
        preferred_category, loyalty_level), or ``None`` if not found.
    """
    logger.info("manager_sql_tool.get_customer_details — user_id=%s", user_id)
    db = SessionLocal()
    try:
        return _user_svc.get_customer_profile(db, user_id)
    finally:
        db.close()


def customer_purchase_history(user_id: str) -> dict:
    """
    Return the complete order history for any customer (manager-only).

    Args:
        user_id: Any customer's identifier.

    Returns:
        Dict with ``user_id``, ``total_orders``, ``total_spent``,
        ``avg_order_value``, and ``orders`` list.
    """
    logger.info(
        "manager_sql_tool.customer_purchase_history — user_id=%s", user_id
    )
    db = SessionLocal()
    try:
        return _order_svc.get_customer_orders_for_manager(db, user_id)
    finally:
        db.close()
