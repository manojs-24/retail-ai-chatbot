"""
Assembles *all* data needed in one call:
  1. KPI summary (SQL / Pandas)
  2. Chart data: monthly sales, category revenue, top products
  3. Table data: low-stock products, top customers
  4. ML outputs: forecast, demand, inventory risk, segments, sentiment
  5. AI Insights: rule-based, data-driven observations (NO LLM)

"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.repositories.order_repository import OrderRepository
from backend.repositories.product_repository import ProductRepository
from backend.repositories.review_repository import ReviewRepository
from backend.repositories.user_repository import UserRepository
from backend.models.models import Order, OrderItem, Product, User, ProductReview

# ML modules
from backend.ml import (
    sales_forecast,
    demand_prediction,
    customer_segmentation,
    inventory_prediction,
    sentiment_analysis,
    product_performance,
)

logger = get_logger(__name__)

_order_repo = OrderRepository()
_product_repo = ProductRepository()
_review_repo = ReviewRepository()
_user_repo = UserRepository()


# Public API

def get_dashboard_data(db: Session) -> dict:
    """
    Returns:
        Dict with keys:

        - ``kpis``          — revenue, orders, customers, products, stock, low_stock
        - ``charts``        — monthly_sales, category_revenue, top_products
        - ``tables``        — low_stock_products, top_customers
        - ``ml``            — forecast, demand, inventory_risk, segments, sentiment, performance
        - ``insights``      — list[str] rule-based business observations
    """
    logger.info("DashboardService.get_dashboard_data — start")

    
    # 1. Raw data loads (ORM → plain Python)
    
    monthly_raw = _order_repo.monthly_sales(db)
    inv_stats = _product_repo.inventory_stats(db)
    low_stock_products = _product_repo.get_low_stock(db)
    top_products_raw = _product_repo.top_selling_with_revenue(db, limit=15)
    categories_raw = _product_repo.category_summary(db)
    sales_raw = _order_repo.sales_summary(db)
    total_customers: int = db.query(User).filter(User.role == "customer").count()

    # All orders + order items for ML
    all_orders = _load_all_orders(db)
    all_order_items = _load_all_order_items(db)
    all_reviews = _load_all_reviews(db)
    all_products_full = _load_all_products_full(db)
    all_customers = _load_all_customers(db)

    # Category revenue
    cat_rev = _load_category_revenue(db)

    # Top customers
    top_customers = _load_top_customers(db)

    
    # 2. KPIs
    
    kpis = {
        "total_revenue": sales_raw["total_revenue"],
        "total_revenue_fmt": _fmt_inr(sales_raw["total_revenue"]),
        "total_orders": sales_raw["total_orders"],
        "avg_order_value": sales_raw["avg_order_value"],
        "total_customers": total_customers,
        "total_products": inv_stats["total_products"],
        "total_stock": inv_stats["total_stock"],
        "low_stock_count": inv_stats["low_stock"],
        "out_of_stock_count": inv_stats["out_of_stock"],
    }

    
    # 3. Chart data
    
    charts = {
        "monthly_sales": _build_monthly_chart(monthly_raw),
        "category_revenue": cat_rev,
        "top_products": top_products_raw[:10],
    }

    
    # 4. Table data
    
    tables = {
        "low_stock_products": [_fmt_low_stock(p) for p in low_stock_products],
        "top_customers": top_customers,
    }

    
    # 5. ML outputs
    
    forecast = sales_forecast.run(monthly_raw)
    demand = demand_prediction.run(top_products_raw)
    inv_risk = inventory_prediction.run(all_products_full, all_order_items)
    segments = customer_segmentation.run(all_orders, all_customers)
    sentiment = sentiment_analysis.run(all_reviews)
    performance = product_performance.run(all_products_full)

    ml = {
        "forecast": forecast,
        "demand": demand,
        "inventory_risk": inv_risk,
        "segments": segments,
        "sentiment": sentiment,
        "performance": performance,
    }

    
    # 6. AI Insights (rule-based, no LLM)
    
    insights = _generate_insights(kpis, ml, charts)

    logger.info("DashboardService.get_dashboard_data — complete")
    return {
        "kpis": kpis,
        "charts": charts,
        "tables": tables,
        "ml": ml,
        "insights": insights,
    }


# Rule-based AI Insights

def _generate_insights(kpis: dict, ml: dict, charts: dict) -> list[str]:
    """Generate concise, data-driven business observations without using an LLM."""
    insights: list[str] = []
    forecast = ml.get("forecast", {})
    inv_risk = ml.get("inventory_risk", {})
    sentiment = ml.get("sentiment", {})
    demand = ml.get("demand", {})
    segments = ml.get("segments", {})

    # Revenue trend
    trend = forecast.get("trend", "stable")
    trend_pct = forecast.get("trend_pct", 0)
    forecast_fmt = forecast.get("forecast_30d_fmt", "N/A")
    if trend == "up":
        insights.append(
            f"📈 Revenue is trending **up {trend_pct:.1f}%** MoM. "
            f"Forecast for the next 30 days: **{forecast_fmt}**."
        )
    elif trend == "down":
        insights.append(
            f"📉 Revenue is trending **down {abs(trend_pct):.1f}%** MoM. "
            f"Forecast for the next 30 days: **{forecast_fmt}** — review pricing or promotions."
        )
    else:
        insights.append(
            f"📊 Revenue is **stable** MoM. "
            f"30-day forecast: **{forecast_fmt}**."
        )

    # Inventory risk
    critical = inv_risk.get("critical_count", 0)
    high = inv_risk.get("high_count", 0)
    if critical > 0:
        insights.append(
            f"🚨 **{critical} product(s)** are at CRITICAL stockout risk (< 7 days). "
            "Immediate restocking required."
        )
    elif high > 0:
        insights.append(
            f"⚠️ **{high} product(s)** have HIGH inventory risk (< 15 days). "
            "Consider placing purchase orders soon."
        )
    elif kpis.get("low_stock_count", 0) > 0:
        insights.append(
            f"📦 **{kpis['low_stock_count']} product(s)** are below reorder level. "
            "Schedule restocking."
        )
    else:
        insights.append("✅ Inventory levels are healthy across all product lines.")

    # Sentiment
    pos_pct = sentiment.get("positive_pct", 0)
    neg_n = sentiment.get("distribution", {}).get("Negative", 0)
    total_rev = sentiment.get("total_reviews", 1)
    if pos_pct >= 70:
        insights.append(
            f"💚 Customer sentiment is strongly positive — "
            f"**{pos_pct:.0f}%** of reviews are positive."
        )
    elif neg_n / max(total_rev, 1) > 0.20:
        insights.append(
            f"🔴 Negative reviews account for "
            f"**{neg_n / max(total_rev, 1) * 100:.0f}%** of all reviews. "
            "Investigate product quality or delivery issues."
        )
    else:
        insights.append(
            f"😊 Sentiment is mostly neutral-to-positive "
            f"(**{pos_pct:.0f}%** positive). Monitor for shifts."
        )

    # Demand highlight
    top_demand = demand.get("high_demand", [])
    if top_demand:
        top_name = top_demand[0].get("product_name", "")
        insights.append(
            f"🔥 Highest-demand product: **{top_name}** — "
            "ensure stock is available to capture demand."
        )

    # Customer segments
    seg_list = segments.get("segments", [])
    champions = next((s for s in seg_list if "Champion" in s.get("segment", "")), None)
    at_risk = next((s for s in seg_list if "At-Risk" in s.get("segment", "")), None)
    if champions:
        insights.append(
            f"🏆 **{champions['customer_count']} Champion customers** "
            f"averaging ₹{champions['avg_spend']:,.0f} in spend — "
            "prioritise retention and loyalty rewards."
        )
    if at_risk:
        insights.append(
            f"⚡ **{at_risk['customer_count']} At-Risk customers** detected. "
            "Consider re-engagement campaigns."
        )

    return insights


# Private data-loading helpers

def _load_all_orders(db: Session) -> list[dict]:
    rows = db.query(Order.user_id, Order.order_date, Order.total_amount).all()
    return [
        {"user_id": r.user_id, "order_date": str(r.order_date), "total_amount": r.total_amount}
        for r in rows
    ]


def _load_all_order_items(db: Session) -> list[dict]:
    rows = db.query(
        OrderItem.product_id, OrderItem.quantity, Order.order_date
    ).join(Order, OrderItem.order_id == Order.order_id).all()
    return [
        {"product_id": r.product_id, "quantity": r.quantity, "order_date": str(r.order_date)}
        for r in rows
    ]


def _load_all_reviews(db: Session) -> list[dict]:
    rows = db.query(
        ProductReview.product_id,
        ProductReview.sentiment,
        ProductReview.rating,
        ProductReview.review_text,
        Product.category,
    ).join(Product, ProductReview.product_id == Product.product_id).all()
    return [
        {
            "product_id": r.product_id,
            "sentiment": r.sentiment,
            "rating": r.rating,
            "review_text": r.review_text or "",
            "category": r.category,
        }
        for r in rows
    ]


def _load_all_products_full(db: Session) -> list[dict]:
    from sqlalchemy import func
    rows = db.query(
        Product.product_id,
        Product.product_name,
        Product.brand,
        Product.category,
        Product.final_price,
        Product.stock_quantity,
        Product.reorder_level,
        Product.rating,
        Product.total_reviews,
        Product.total_sold,
    ).filter(Product.status == "Active").all()
    # Merge with actual units_sold from order_items
    sold_rows = db.query(
        OrderItem.product_id,
        func.sum(OrderItem.quantity).label("units_sold"),
        func.sum(OrderItem.subtotal).label("revenue"),
    ).group_by(OrderItem.product_id).all()
    sold_map = {r.product_id: {"units_sold": int(r.units_sold), "revenue": float(r.revenue)} for r in sold_rows}

    return [
        {
            "product_id": r.product_id,
            "product_name": r.product_name,
            "brand": r.brand,
            "category": r.category,
            "final_price": float(r.final_price or 0),
            "stock_quantity": int(r.stock_quantity or 0),
            "reorder_level": int(r.reorder_level or 0),
            "rating": float(r.rating or 3.0),
            "total_reviews": int(r.total_reviews or 0),
            "units_sold": sold_map.get(r.product_id, {}).get("units_sold", 0),
            "revenue": sold_map.get(r.product_id, {}).get("revenue", 0.0),
        }
        for r in rows
    ]


def _load_all_customers(db: Session) -> list[dict]:
    rows = db.query(User.user_id).filter(User.role == "customer").all()
    return [{"user_id": r.user_id} for r in rows]


def _load_category_revenue(db: Session) -> list[dict]:
    from sqlalchemy import func
    rows = db.query(
        Product.category,
        func.round(func.sum(OrderItem.subtotal), 2).label("revenue"),
    ).join(OrderItem, Product.product_id == OrderItem.product_id).group_by(Product.category).all()
    return [
        {"category": r.category, "revenue": float(r.revenue or 0)}
        for r in sorted(rows, key=lambda x: x[1], reverse=True)
    ]


def _load_top_customers(db: Session, limit: int = 10) -> list[dict]:
    from sqlalchemy import func
    rows = db.query(
        User.user_id,
        User.full_name,
        User.city,
        User.loyalty_level,
        func.count(Order.order_id).label("order_count"),
        func.round(func.sum(Order.total_amount), 2).label("total_spent"),
    ).join(Order, User.user_id == Order.user_id).filter(User.role == "customer").group_by(User.user_id).order_by(
        func.sum(Order.total_amount).desc()
    ).limit(limit).all()
    return [
        {
            "user_id": r.user_id,
            "full_name": r.full_name,
            "city": r.city,
            "loyalty_level": r.loyalty_level,
            "order_count": int(r.order_count),
            "total_spent": float(r.total_spent or 0),
        }
        for r in rows
    ]


def _build_monthly_chart(monthly_raw: list[dict]) -> list[dict]:
    data = sorted(monthly_raw, key=lambda r: (r["year"], r["month"]))
    return [
        {
            "label": f"{int(r['year'])}-{int(r['month']):02d}",
            "revenue": r["revenue"],
            "order_count": r["order_count"],
        }
        for r in data[-18:]  # Last 18 months
    ]


def _fmt_low_stock(p: Product) -> dict:
    return {
        "product_id": p.product_id,
        "product_name": p.product_name,
        "brand": p.brand,
        "category": p.category,
        "stock_quantity": p.stock_quantity,
        "reorder_level": p.reorder_level,
        "status": "Out of Stock" if p.stock_quantity == 0 else "Low Stock",
    }


def _fmt_inr(value: float) -> str:
    if value >= 1_00_00_000:
        return f"₹{value / 1_00_00_000:.2f} Cr"
    if value >= 1_00_000:
        return f"₹{value / 1_00_000:.1f} L"
    return f"₹{value:,.0f}"
