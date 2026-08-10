"""
Customer Dashboard Service
==========================
Single entry-point for the Customer Dashboard page.

Assembles every data bundle the dashboard needs in one call:

1. Customer profile & welcome card data
2. KPI cards  — orders, spending, reward points, wishlist placeholder
3. Chart data — monthly spending trend, category breakdown, order status
4. Table data — recent orders, active orders, recent reviews
5. Shopping insights — rule-based, data-driven observations (NO LLM)

Separation of concerns
-----------------------
This service is completely independent of the chatbot so the LangGraph
customer SQL node can continue to import individual repositories without
triggering a full dashboard rebuild.
"""

from __future__ import annotations

import logging
from calendar import month_abbr
from datetime import date

import pandas as pd
from sqlalchemy import extract, func
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.models import Order, OrderItem, Product, ProductReview, User
from backend.repositories.order_repository import OrderRepository
from backend.repositories.product_repository import ProductRepository
from backend.repositories.review_repository import ReviewRepository
from backend.repositories.user_repository import UserRepository

logger = get_logger(__name__)

_order_repo = OrderRepository()
_product_repo = ProductRepository()
_review_repo = ReviewRepository()
_user_repo = UserRepository()

# Loyalty points multiplier per tier
_LOYALTY_POINTS_RATE: dict[str, float] = {
    "Bronze":   1.0,
    "Silver":   1.5,
    "Gold":     2.0,
    "Platinum": 3.0,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dashboard_data(db: Session, user_id: str) -> dict:
    """
    Collect and return every data bundle required by the Customer Dashboard.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Authenticated customer identifier from session state.

    Returns:
        Dict with keys:

        - ``profile``   — welcome card data
        - ``kpis``      — order count, spend, reward points, wishlist
        - ``charts``    — monthly spending, category breakdown, order status
        - ``tables``    — recent orders, active orders, recent reviews
        - ``insights``  — list[str] rule-based shopping observations
    """
    logger.info("CustomerDashboardService.get_dashboard_data — user_id=%s", user_id)

    user = _user_repo.get_by_id(db, user_id)
    if user is None:
        logger.error("User not found — user_id=%s", user_id)
        return {}

    # ------------------------------------------------------------------
    # Raw data loads
    # ------------------------------------------------------------------
    all_orders     = _order_repo.get_orders_by_user(db, user_id)
    recent_orders  = all_orders[:5]
    active_orders  = _get_active_orders(db, user_id)
    recent_reviews = _get_recent_reviews(db, user_id, limit=5)
    order_stats    = _order_repo.customer_order_stats(db, user_id)
    monthly_spend  = _get_monthly_spending(db, user_id)
    cat_spend      = _get_category_spending(db, user_id)
    status_dist    = _get_order_status_distribution(db, user_id)
    total_savings  = _get_total_savings(db, user_id)
    top_brand      = _get_top_brand(db, user_id)
    this_month_cnt = _get_orders_this_month(db, user_id)

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------
    profile = {
        "user_id":            user.user_id,
        "full_name":          user.full_name,
        "email":              user.email,
        "city":               user.city or "",
        "state":              user.state or "",
        "join_date":          str(user.join_date) if user.join_date else "N/A",
        "loyalty_level":      user.loyalty_level or "Bronze",
        "preferred_category": user.preferred_category or "General",
        "gender":             user.gender or "",
        "age":                user.age or 0,
    }

    # ------------------------------------------------------------------
    # KPIs
    # ------------------------------------------------------------------
    total_spent = order_stats["total_spent"]
    loyalty_rate = _LOYALTY_POINTS_RATE.get(user.loyalty_level or "Bronze", 1.0)
    reward_points = int(total_spent / 100 * loyalty_rate)   # ₹100 = base points

    kpis = {
        "total_orders":    order_stats["total_orders"],
        "total_spent":     total_spent,
        "total_spent_fmt": _fmt_inr(total_spent),
        "avg_order_value": order_stats["avg_order_value"],
        "reward_points":   reward_points,
        "wishlist_count":  0,           # Placeholder — wishlist not yet implemented
        "total_savings":   total_savings,
        "total_savings_fmt": _fmt_inr(total_savings),
    }

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    charts = {
        "monthly_spending": monthly_spend,
        "category_spending": cat_spend,
        "order_status": status_dist,
    }

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    tables = {
        "recent_orders":  [_fmt_order(o) for o in recent_orders],
        "active_orders":  [_fmt_order(o) for o in active_orders],
        "recent_reviews": recent_reviews,
    }

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------
    insights = _generate_insights(
        profile=profile,
        kpis=kpis,
        cat_spend=cat_spend,
        top_brand=top_brand,
        this_month_cnt=this_month_cnt,
        order_stats=order_stats,
    )

    return {
        "profile":  profile,
        "kpis":     kpis,
        "charts":   charts,
        "tables":   tables,
        "insights": insights,
    }


# ---------------------------------------------------------------------------
# Individual query helpers (usable independently by other services)
# ---------------------------------------------------------------------------

def get_customer_summary(db: Session, user_id: str) -> dict:
    """
    Return a lightweight customer profile + order stats dict.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.

    Returns:
        Dict with profile fields and order statistics.
    """
    user = _user_repo.get_by_id(db, user_id)
    if user is None:
        return {}
    stats = _order_repo.customer_order_stats(db, user_id)
    return {
        "user_id":            user.user_id,
        "full_name":          user.full_name,
        "loyalty_level":      user.loyalty_level,
        "preferred_category": user.preferred_category,
        "join_date":          str(user.join_date) if user.join_date else None,
        **stats,
    }


def get_recent_orders(db: Session, user_id: str, limit: int = 5) -> list[dict]:
    """
    Return the *limit* most recent orders for *user_id*.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.
        limit:   Number of orders to return.

    Returns:
        List of order dicts.
    """
    orders = _order_repo.get_orders_by_user(db, user_id, limit=limit)
    return [_fmt_order(o) for o in orders]


def get_active_orders(db: Session, user_id: str) -> list[dict]:
    """
    Return all in-progress orders (not yet Delivered or Cancelled).

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.

    Returns:
        List of active order dicts.
    """
    return [_fmt_order(o) for o in _get_active_orders(db, user_id)]


def get_total_spending(db: Session, user_id: str) -> dict:
    """
    Return total and average spending for *user_id*.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.

    Returns:
        Dict with ``total_spent``, ``avg_order_value``, ``total_orders``.
    """
    return _order_repo.customer_order_stats(db, user_id)


def get_favorite_categories(db: Session, user_id: str) -> list[dict]:
    """
    Return the customer's purchase spending grouped by product category.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.

    Returns:
        List of dicts with ``category`` and ``spent``, sorted by spend desc.
    """
    return _get_category_spending(db, user_id)


def get_recent_reviews(db: Session, user_id: str, limit: int = 5) -> list[dict]:
    """
    Return the *limit* most recent reviews written by *user_id*.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.
        limit:   Max reviews to return.

    Returns:
        List of review dicts.
    """
    return _get_recent_reviews(db, user_id, limit)


def get_reward_summary(db: Session, user_id: str) -> dict:
    """
    Compute reward point balance for *user_id* based on total spend and loyalty tier.

    Args:
        db:      Active SQLAlchemy session.
        user_id: Customer identifier.

    Returns:
        Dict with ``loyalty_level``, ``reward_points``, ``points_rate``.
    """
    user = _user_repo.get_by_id(db, user_id)
    if user is None:
        return {"loyalty_level": "Bronze", "reward_points": 0, "points_rate": 1.0}
    stats = _order_repo.customer_order_stats(db, user_id)
    rate = _LOYALTY_POINTS_RATE.get(user.loyalty_level or "Bronze", 1.0)
    points = int(stats["total_spent"] / 100 * rate)
    return {
        "loyalty_level": user.loyalty_level,
        "reward_points": points,
        "points_rate":   rate,
        "total_spent":   stats["total_spent"],
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_active_orders(db: Session, user_id: str) -> list[Order]:
    return (
        db.query(Order)
        .filter(
            Order.user_id == user_id,
            Order.delivery_status.notin_(["Delivered", "Cancelled"]),
        )
        .order_by(Order.order_date.desc())
        .all()
    )


def _get_recent_reviews(db: Session, user_id: str, limit: int) -> list[dict]:
    rows = (
        db.query(
            ProductReview.review_id,
            ProductReview.product_id,
            Product.product_name,
            Product.brand,
            ProductReview.rating,
            ProductReview.review_title,
            ProductReview.review_text,
            ProductReview.sentiment,
            ProductReview.review_date,
        )
        .join(Product, ProductReview.product_id == Product.product_id)
        .filter(ProductReview.user_id == user_id)
        .order_by(ProductReview.review_date.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "review_id":    r.review_id,
            "product_id":   r.product_id,
            "product_name": r.product_name,
            "brand":        r.brand,
            "rating":       r.rating,
            "title":        r.review_title or "",
            "text":         (r.review_text or "")[:120],
            "sentiment":    r.sentiment or "",
            "review_date":  str(r.review_date) if r.review_date else "",
        }
        for r in rows
    ]


def _get_monthly_spending(db: Session, user_id: str) -> list[dict]:
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


def _get_category_spending(db: Session, user_id: str) -> list[dict]:
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
    return [
        {"category": r.category, "spent": float(r.spent or 0)}
        for r in rows
    ]


def _get_order_status_distribution(db: Session, user_id: str) -> list[dict]:
    rows = (
        db.query(
            Order.delivery_status,
            func.count(Order.order_id).label("count"),
        )
        .filter(Order.user_id == user_id)
        .group_by(Order.delivery_status)
        .all()
    )
    return [
        {"status": r.delivery_status or "Unknown", "count": int(r.count)}
        for r in rows
    ]


def _get_total_savings(db: Session, user_id: str) -> float:
    val = (
        db.query(func.coalesce(func.sum(OrderItem.discount), 0))
        .join(Order, OrderItem.order_id == Order.order_id)
        .filter(Order.user_id == user_id)
        .scalar()
    )
    return round(float(val), 2)


def _get_top_brand(db: Session, user_id: str) -> str:
    row = (
        db.query(
            Product.brand,
            func.sum(OrderItem.quantity).label("qty"),
        )
        .join(OrderItem, Product.product_id == OrderItem.product_id)
        .join(Order, OrderItem.order_id == Order.order_id)
        .filter(Order.user_id == user_id)
        .group_by(Product.brand)
        .order_by(func.sum(OrderItem.quantity).desc())
        .first()
    )
    return row.brand if row else "N/A"


def _get_orders_this_month(db: Session, user_id: str) -> int:
    today = date.today()
    return (
        db.query(func.count(Order.order_id))
        .filter(
            Order.user_id == user_id,
            extract("year",  Order.order_date) == today.year,
            extract("month", Order.order_date) == today.month,
        )
        .scalar()
    ) or 0


def _fmt_order(o: Order) -> dict:
    return {
        "order_id":       o.order_id,
        "order_date":     str(o.order_date) if o.order_date else "",
        "total_amount":   o.total_amount,
        "payment_method": o.payment_method or "",
        "payment_status": o.payment_status or "",
        "delivery_status": o.delivery_status or "",
        "delivery_date":  str(o.delivery_date) if o.delivery_date else "—",
    }


def _fmt_inr(value: float) -> str:
    if value >= 1_00_00_000:
        return f"₹{value / 1_00_00_000:.2f} Cr"
    if value >= 1_00_000:
        return f"₹{value / 1_00_000:.1f} L"
    return f"₹{value:,.0f}"


# ---------------------------------------------------------------------------
# Rule-based insights
# ---------------------------------------------------------------------------

def _generate_insights(
    profile: dict,
    kpis: dict,
    cat_spend: list[dict],
    top_brand: str,
    this_month_cnt: int,
    order_stats: dict,
) -> list[str]:
    """
    Generate concise, personalised shopping insights without using an LLM.

    Args:
        profile:        Customer profile dict.
        kpis:           KPI dict from the dashboard data.
        cat_spend:      Category spending list (sorted desc by spend).
        top_brand:      Most-purchased brand name.
        this_month_cnt: Number of orders placed this calendar month.
        order_stats:    Dict from OrderRepository.customer_order_stats.

    Returns:
        List of Markdown-formatted insight strings.
    """
    insights: list[str] = []
    loyalty = profile.get("loyalty_level", "Bronze")
    pref_cat = profile.get("preferred_category", "General")

    # Favourite category
    if cat_spend:
        top_cat = cat_spend[0]["category"]
        top_spent = cat_spend[0]["spent"]
        insights.append(
            f"🛍️ Your favourite shopping category is **{top_cat}** "
            f"with **{_fmt_inr(top_spent)}** spent."
        )

    # Average order value
    avg = order_stats.get("avg_order_value", 0)
    if avg > 0:
        insights.append(
            f"💳 Your average order value is **{_fmt_inr(avg)}** — "
            + ("you tend to make premium purchases!" if avg > 50_000
               else "great value shopping!")
        )

    # Total savings
    savings = kpis.get("total_savings", 0)
    if savings > 0:
        insights.append(
            f"💰 You've saved **{_fmt_inr(savings)}** through discounts so far. "
            "Keep an eye out for more deals!"
        )

    # Most purchased brand
    if top_brand and top_brand != "N/A":
        insights.append(
            f"🏷️ Your most-purchased brand is **{top_brand}**. "
            "Check out their latest arrivals!"
        )

    # Orders this month
    if this_month_cnt > 0:
        insights.append(
            f"📦 You've placed **{this_month_cnt} order(s)** this month. "
            + ("You're on a shopping spree! 🎉" if this_month_cnt >= 3
               else "Stay tuned for new arrivals.")
        )
    else:
        insights.append(
            "📭 No orders placed yet this month. "
            "Browse our latest deals and recommendations below!"
        )

    # Loyalty progress
    next_tier = {"Bronze": "Silver", "Silver": "Gold", "Gold": "Platinum"}.get(loyalty)
    if next_tier:
        insights.append(
            f"🏆 You're a **{loyalty}** member. "
            f"Keep shopping to reach **{next_tier}** tier and unlock better rewards!"
        )
    else:
        insights.append(
            f"🏆 You're a **Platinum** member — our highest loyalty tier. "
            "Enjoy 3× reward points on every purchase!"
        )

    return insights
