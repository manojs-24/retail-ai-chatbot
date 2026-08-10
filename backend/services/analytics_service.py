"""
Analytics Service
=================
Service layer for store-level analytics operations (manager-facing).

Orchestrates calls to multiple repositories and assembles structured
summaries ready for LLM consumption or dashboard display.

Future phases will add:
- Trend detection, YoY comparisons
- Customer segmentation (RFM)
- Cohort analysis
- Predictive inventory replenishment
"""

from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.repositories.product_repository import ProductRepository
from backend.repositories.order_repository import OrderRepository
from backend.repositories.user_repository import UserRepository

logger = get_logger(__name__)


class AnalyticsService:
    """Service layer for store analytics operations."""

    def __init__(self) -> None:
        self._product_repo = ProductRepository()
        self._order_repo = OrderRepository()
        self._user_repo = UserRepository()

    # ------------------------------------------------------------------
    # Inventory analytics
    # ------------------------------------------------------------------

    def get_inventory_summary(self, db: Session) -> dict:
        """
        Return a high-level inventory health snapshot.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            Dict with ``total_products``, ``total_stock``,
            ``out_of_stock``, ``low_stock``, and ``low_stock_products``
            list (top 10 most critical items).
        """
        logger.info("AnalyticsService.get_inventory_summary")
        stats = self._product_repo.inventory_stats(db)
        low_stock = self._product_repo.get_low_stock(db)

        low_stock_list = [
            {
                "product_id": p.product_id,
                "product_name": p.product_name,
                "brand": p.brand,
                "category": p.category,
                "stock_quantity": p.stock_quantity,
                "reorder_level": p.reorder_level,
            }
            for p in low_stock[:10]
        ]

        return {**stats, "low_stock_products": low_stock_list}

    def get_low_stock_products(self, db: Session) -> dict:
        """
        Return all low-stock and out-of-stock products.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            Dict with ``low_stock_count``, ``out_of_stock_count``,
            and ``products`` list.
        """
        logger.info("AnalyticsService.get_low_stock_products")
        low_stock = self._product_repo.get_low_stock(db)
        out_of_stock = self._product_repo.get_out_of_stock(db)

        def _fmt(p):
            return {
                "product_id": p.product_id,
                "product_name": p.product_name,
                "brand": p.brand,
                "category": p.category,
                "stock_quantity": p.stock_quantity,
                "reorder_level": p.reorder_level,
                "status": "Out of Stock" if p.stock_quantity == 0 else "Low Stock",
            }

        return {
            "low_stock_count": len(low_stock),
            "out_of_stock_count": len(out_of_stock),
            "products": [_fmt(p) for p in low_stock],
        }

    def get_top_selling_products(self, db: Session, limit: int = 10) -> dict:
        """
        Return the top-selling products by units sold with revenue figures.

        Args:
            db:    Active SQLAlchemy session.
            limit: Number of products to return.

        Returns:
            Dict with ``limit``, ``count``, and ``products`` list.
        """
        logger.info(
            "AnalyticsService.get_top_selling_products — limit=%d", limit
        )
        products = self._product_repo.top_selling_with_revenue(db, limit=limit)
        return {"limit": limit, "count": len(products), "products": products}

    def get_category_summary(self, db: Session) -> dict:
        """
        Return product and stock counts grouped by category.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            Dict with ``category_count`` and ``categories`` list.
        """
        logger.info("AnalyticsService.get_category_summary")
        categories = self._product_repo.category_summary(db)
        return {"category_count": len(categories), "categories": categories}

    # ------------------------------------------------------------------
    # Sales analytics
    # ------------------------------------------------------------------

    def get_sales_summary(self, db: Session) -> dict:
        """
        Return store-wide sales summary.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            Dict with ``total_revenue``, ``total_orders``,
            ``avg_order_value``.
        """
        logger.info("AnalyticsService.get_sales_summary")
        return self._order_repo.sales_summary(db)

    def get_monthly_sales(self, db: Session) -> dict:
        """
        Return monthly sales grouped by year-month, newest first.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            Dict with ``total_months`` and ``months`` list.
        """
        logger.info("AnalyticsService.get_monthly_sales")
        months = self._order_repo.monthly_sales(db)
        return {"total_months": len(months), "months": months}

    def get_full_sales_analytics(self, db: Session) -> dict:
        """
        Return a comprehensive sales analytics package.

        Combines: overall summary, monthly breakdown, top products by units
        and revenue, and revenue by category.  All calculations are done
        in SQLAlchemy + Pandas — the LLM only narrates the result.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            Dict with keys:
            - ``total_revenue``, ``total_orders``, ``avg_order_value``
            - ``monthly_breakdown`` — last 12 months, newest first
            - ``top_selling_products`` — top 10 by units sold
            - ``top_revenue_products`` — top 10 by revenue
            - ``revenue_by_category``  — all categories ranked by revenue
            - ``sales_trend``          — "up" | "down" | "stable" + pct
        """
        import pandas as pd

        logger.info("AnalyticsService.get_full_sales_analytics")

        # Core KPIs
        summary = self._order_repo.sales_summary(db)

        # Monthly breakdown (last 12 months, newest first)
        all_months = self._order_repo.monthly_sales(db)
        monthly_12 = all_months[:12]

        # Sales trend: compare last two months
        if len(all_months) >= 2:
            last_rev = all_months[0]["revenue"]
            prev_rev = all_months[1]["revenue"]
            trend_pct = ((last_rev - prev_rev) / prev_rev * 100) if prev_rev else 0.0
            trend = "up" if trend_pct > 2 else ("down" if trend_pct < -2 else "stable")
        else:
            trend_pct = 0.0
            trend = "stable"

        # Top selling by units
        top_units = self._product_repo.top_selling_with_revenue(db, limit=10)

        # Top revenue products
        top_revenue = self._order_repo.top_revenue_products(db, limit=10)

        # Revenue by category
        cat_revenue = self._order_repo.revenue_by_category(db)

        return {
            **summary,
            "monthly_breakdown": monthly_12,
            "sales_trend": {"direction": trend, "pct_change": round(trend_pct, 1)},
            "top_selling_products": top_units,
            "top_revenue_products": top_revenue,
            "revenue_by_category": cat_revenue,
        }

    def get_customer_analytics(self, db: Session) -> dict:
        """
        Return customer analytics: highest-spending customers and
        customer count by loyalty tier.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            Dict with ``highest_spenders`` list and ``total_customers``.
        """
        logger.info("AnalyticsService.get_customer_analytics")

        top_spenders = self._order_repo.highest_spending_customers(db, limit=10)
        total_customers = len(self._user_repo.get_all_customers(db))

        return {
            "total_customers": total_customers,
            "highest_spenders": top_spenders,
        }
