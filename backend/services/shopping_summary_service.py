"""
Shopping Summary Service
========================
Focused service for computing personalised shopping statistics for a
single customer.  All functions are thin wrappers around ORM queries —
no Pandas required (aggregations are done in SQLite via SQLAlchemy).

These functions are intentionally kept separate from the main dashboard
service so the LangGraph customer SQL node can call them independently
without triggering a full dashboard rebuild.

Functions
---------
- get_spending_by_month(user_id)    — monthly spending time-series
- get_spending_by_category(user_id) — category breakdown
- get_brand_summary(user_id)        — most-purchased brands
- get_payment_method_summary(user_id) — payment method distribution
- get_delivery_status_summary(user_id) — delivery status counts
- get_savings_summary(user_id)      — total discount savings
- get_trending_products()           — store-wide trending (top-sold)
"""

from __future__ import annotations

import logging
from calendar import month_abbr

from sqlalchemy import extract, func
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.models import Order, OrderItem, Product

logger = get_logger(__name__)


def get_spending_by_month(db: Session, user_id: str) -> list[dict]:
    """
    Return the customer's total spending grouped by calendar month.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.

    Returns:
        List of dicts with ``label`` (e.g. ``"Jan 2025"``), ``spent``,
        ``order_count``.  Sorted chronologically (oldest first).
    """
    logger.debug("ShoppingSummaryService.get_spending_by_month — user_id=%s", user_id)
    rows = (
        db.query(
            extract("year",  Order.order_date).label("year"),
            extract("month", Order.order_date).label("month"),
            func.round(func.sum(Order.total_amount), 2).label("spent"),
            func.count(Order.order_id).label("order_count"),
        )
        .filter(Order.user_id == user_id)
        .group_by("year", "month")
        .order_by(
            extract("year",  Order.order_date).asc(),
            extract("month", Order.order_date).asc(),
        )
        .all()
    )
    return [
        {
            "label":       f"{month_abbr[int(r.month)]} {int(r.year)}",
            "spent":       float(r.spent or 0),
            "order_count": int(r.order_count),
        }
        for r in rows
    ]


def get_spending_by_category(db: Session, user_id: str) -> list[dict]:
    """
    Return the customer's total spending grouped by product category.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.

    Returns:
        List of dicts with ``category`` and ``spent``, sorted desc by spend.
    """
    logger.debug("ShoppingSummaryService.get_spending_by_category — user_id=%s", user_id)
    rows = (
        db.query(
            Product.category,
            func.round(func.sum(OrderItem.subtotal), 2).label("spent"),
        )
        .join(OrderItem, Product.product_id == OrderItem.product_id)
        .join(Order, OrderItem.order_id == Order.order_id)
        .filter(Order.user_id == user_id)
        .group_by(Product.category)
        .order_by(func.sum(OrderItem.subtotal).desc())
        .all()
    )
    return [{"category": r.category, "spent": float(r.spent or 0)} for r in rows]


def get_brand_summary(db: Session, user_id: str, limit: int = 10) -> list[dict]:
    """
    Return the customer's most-purchased brands by total units ordered.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.
        limit:   Number of brands to return.

    Returns:
        List of dicts with ``brand``, ``units_bought``, ``total_spent``.
    """
    logger.debug("ShoppingSummaryService.get_brand_summary — user_id=%s", user_id)
    rows = (
        db.query(
            Product.brand,
            func.sum(OrderItem.quantity).label("units_bought"),
            func.round(func.sum(OrderItem.subtotal), 2).label("total_spent"),
        )
        .join(OrderItem, Product.product_id == OrderItem.product_id)
        .join(Order, OrderItem.order_id == Order.order_id)
        .filter(Order.user_id == user_id)
        .group_by(Product.brand)
        .order_by(func.sum(OrderItem.quantity).desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "brand":       r.brand,
            "units_bought": int(r.units_bought or 0),
            "total_spent":  float(r.total_spent or 0),
        }
        for r in rows
    ]


def get_payment_method_summary(db: Session, user_id: str) -> list[dict]:
    """
    Return a breakdown of orders by payment method for *user_id*.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.

    Returns:
        List of dicts with ``payment_method`` and ``count``.
    """
    logger.debug("ShoppingSummaryService.get_payment_method_summary — user_id=%s", user_id)
    rows = (
        db.query(
            Order.payment_method,
            func.count(Order.order_id).label("count"),
        )
        .filter(Order.user_id == user_id)
        .group_by(Order.payment_method)
        .order_by(func.count(Order.order_id).desc())
        .all()
    )
    return [{"payment_method": r.payment_method or "Unknown", "count": int(r.count)} for r in rows]


def get_delivery_status_summary(db: Session, user_id: str) -> list[dict]:
    """
    Return a count of orders by delivery status for *user_id*.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.

    Returns:
        List of dicts with ``status`` and ``count``.
    """
    logger.debug("ShoppingSummaryService.get_delivery_status_summary — user_id=%s", user_id)
    rows = (
        db.query(
            Order.delivery_status,
            func.count(Order.order_id).label("count"),
        )
        .filter(Order.user_id == user_id)
        .group_by(Order.delivery_status)
        .all()
    )
    return [{"status": r.delivery_status or "Unknown", "count": int(r.count)} for r in rows]


def get_savings_summary(db: Session, user_id: str) -> dict:
    """
    Return total discount savings for *user_id* across all orders.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.

    Returns:
        Dict with ``total_savings`` (float).
    """
    logger.debug("ShoppingSummaryService.get_savings_summary — user_id=%s", user_id)
    val = (
        db.query(func.coalesce(func.sum(OrderItem.discount), 0))
        .join(Order, OrderItem.order_id == Order.order_id)
        .filter(Order.user_id == user_id)
        .scalar()
    )
    return {"total_savings": round(float(val), 2)}


def get_trending_products(db: Session, limit: int = 10) -> list[dict]:
    """
    Return the top-selling active products across the entire store.

    This is a store-wide query — no user_id filter.

    Args:
        db:    Active SQLAlchemy session.
        limit: Number of products to return.

    Returns:
        List of dicts with product fields and ``units_sold``.
    """
    logger.debug("ShoppingSummaryService.get_trending_products — limit=%d", limit)
    rows = (
        db.query(
            Product.product_id,
            Product.product_name,
            Product.brand,
            Product.category,
            Product.final_price,
            Product.rating,
            Product.discount_percentage,
            Product.image_url,
            func.sum(OrderItem.quantity).label("units_sold"),
        )
        .join(OrderItem, Product.product_id == OrderItem.product_id)
        .filter(Product.status == "Active")
        .group_by(Product.product_id)
        .order_by(func.sum(OrderItem.quantity).desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "product_id":           r.product_id,
            "product_name":         r.product_name,
            "brand":                r.brand,
            "category":             r.category,
            "final_price":          float(r.final_price or 0),
            "rating":               float(r.rating or 0),
            "discount_percentage":  float(r.discount_percentage or 0),
            "image_url":            r.image_url or "",
            "units_sold":           int(r.units_sold or 0),
        }
        for r in rows
    ]
