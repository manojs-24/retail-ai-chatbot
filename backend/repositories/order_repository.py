"""
Order Repository
================
Pure data-access layer for the ``orders`` and ``order_items`` tables.

No business logic lives here.
"""

from __future__ import annotations

import logging
from sqlalchemy import extract, func
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.models import Order, OrderItem, Product

logger = get_logger(__name__)


class OrderRepository:
    """
    Data-access layer for :class:`~backend.models.models.Order` and
    :class:`~backend.models.models.OrderItem` records.
    """

    # ------------------------------------------------------------------
    # Order queries
    # ------------------------------------------------------------------

    def get_orders_by_user(
        self,
        db: Session,
        user_id: str,
        limit: int | None = None,
    ) -> list[Order]:
        """
        Return all orders belonging to *user_id*, sorted newest first.

        Args:
            db:      Active SQLAlchemy session.
            user_id: Customer identifier.
            limit:   If provided, return only the *limit* most recent orders.

        Returns:
            List of :class:`~backend.models.models.Order` objects.
        """
        logger.debug(
            "OrderRepository.get_orders_by_user — user_id=%s limit=%s",
            user_id, limit,
        )
        q = (
            db.query(Order)
            .filter(Order.user_id == user_id)
            .order_by(Order.order_date.desc())
        )
        if limit is not None:
            q = q.limit(limit)
        return q.all()

    def get_order_by_id(self, db: Session, order_id: str) -> Order | None:
        """
        Fetch a single order by primary key.

        Args:
            db:       Active SQLAlchemy session.
            order_id: Order identifier (e.g. ``"ORD00001"``).

        Returns:
            :class:`~backend.models.models.Order` object, or ``None``.
        """
        logger.debug("OrderRepository.get_order_by_id — order_id=%s", order_id)
        return db.query(Order).filter(Order.order_id == order_id).first()

    # ------------------------------------------------------------------
    # Order item queries
    # ------------------------------------------------------------------

    def get_items_with_products(
        self,
        db: Session,
        order_id: str,
    ) -> list[tuple[OrderItem, Product]]:
        """
        Return all line items for *order_id* joined with their product rows.

        Args:
            db:       Active SQLAlchemy session.
            order_id: Order identifier.

        Returns:
            List of ``(OrderItem, Product)`` tuples.
        """
        logger.debug(
            "OrderRepository.get_items_with_products — order_id=%s", order_id
        )
        return (
            db.query(OrderItem, Product)
            .join(Product, OrderItem.product_id == Product.product_id)
            .filter(OrderItem.order_id == order_id)
            .all()
        )

    # ------------------------------------------------------------------
    # Aggregate / analytics queries
    # ------------------------------------------------------------------

    def sales_summary(self, db: Session) -> dict:
        """
        Compute store-wide sales statistics.

        Returns:
            Dict with keys ``total_revenue``, ``total_orders``,
            ``avg_order_value``.
        """
        logger.debug("OrderRepository.sales_summary")
        total_revenue = (
            db.query(func.coalesce(func.sum(Order.total_amount), 0)).scalar()
        )
        total_orders: int = db.query(func.count(Order.order_id)).scalar()
        avg_order_value = (
            db.query(func.coalesce(func.avg(Order.total_amount), 0)).scalar()
        )
        return {
            "total_revenue": round(float(total_revenue), 2),
            "total_orders": total_orders,
            "avg_order_value": round(float(avg_order_value), 2),
        }

    def monthly_sales(self, db: Session) -> list[dict]:
        """
        Group orders by calendar year-month, newest first.

        Returns:
            List of dicts with keys ``year``, ``month``, ``order_count``,
            ``revenue``.
        """
        logger.debug("OrderRepository.monthly_sales")
        rows = (
            db.query(
                extract("year", Order.order_date).label("year"),
                extract("month", Order.order_date).label("month"),
                func.count(Order.order_id).label("order_count"),
                func.round(func.sum(Order.total_amount), 2).label("revenue"),
            )
            .group_by("year", "month")
            .order_by(
                extract("year", Order.order_date).desc(),
                extract("month", Order.order_date).desc(),
            )
            .all()
        )
        return [
            {
                "year": int(row.year),
                "month": int(row.month),
                "order_count": int(row.order_count),
                "revenue": float(row.revenue or 0),
            }
            for row in rows
        ]

    def get_order_details_for_manager(
        self,
        db: Session,
        order_id: str,
    ) -> dict | None:
        """
        Return full order details including line items — manager view (no ownership check).

        Args:
            db:       Active SQLAlchemy session.
            order_id: Order identifier (e.g. ``"ORD00123"``).

        Returns:
            Dict with order header fields, ``customer_id``, and ``items`` list,
            or ``None`` if the order does not exist.
        """
        logger.debug(
            "OrderRepository.get_order_details_for_manager — order_id=%s", order_id
        )
        order = self.get_order_by_id(db, order_id)
        if order is None:
            return None
        items = self.get_items_with_products(db, order_id)
        return {
            "order_id": order.order_id,
            "customer_id": order.user_id,
            "order_date": str(order.order_date) if order.order_date else None,
            "total_amount": order.total_amount,
            "payment_method": order.payment_method,
            "payment_status": order.payment_status,
            "delivery_status": order.delivery_status,
            "shipping_address": order.shipping_address,
            "delivery_date": str(order.delivery_date) if order.delivery_date else None,
            "items": [
                {
                    "product_id": product.product_id,
                    "product_name": product.product_name,
                    "brand": product.brand,
                    "category": product.category,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "discount": item.discount,
                    "subtotal": item.subtotal,
                }
                for item, product in items
            ],
        }

    def revenue_by_category(self, db: Session) -> list[dict]:
        """
        Return total revenue and order count grouped by product category.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            List of dicts with ``category``, ``revenue``, ``order_count``,
            sorted by revenue descending.
        """
        from backend.models.models import Product  # avoid circular at module level
        logger.debug("OrderRepository.revenue_by_category")
        rows = (
            db.query(
                Product.category,
                func.round(func.sum(OrderItem.subtotal), 2).label("revenue"),
                func.count(func.distinct(OrderItem.order_id)).label("order_count"),
                func.sum(OrderItem.quantity).label("units_sold"),
            )
            .join(Product, OrderItem.product_id == Product.product_id)
            .group_by(Product.category)
            .order_by(func.sum(OrderItem.subtotal).desc())
            .all()
        )
        return [
            {
                "category": row.category,
                "revenue": float(row.revenue or 0),
                "order_count": int(row.order_count or 0),
                "units_sold": int(row.units_sold or 0),
            }
            for row in rows
        ]

    def top_revenue_products(self, db: Session, limit: int = 10) -> list[dict]:
        """
        Return top products ranked by total revenue generated (not just units).

        Args:
            db:    Active SQLAlchemy session.
            limit: Maximum rows to return.

        Returns:
            List of dicts with product details plus ``revenue`` and ``units_sold``.
        """
        from backend.models.models import Product
        logger.debug("OrderRepository.top_revenue_products — limit=%d", limit)
        rows = (
            db.query(
                Product.product_id,
                Product.product_name,
                Product.brand,
                Product.category,
                Product.final_price,
                func.round(func.sum(OrderItem.subtotal), 2).label("revenue"),
                func.sum(OrderItem.quantity).label("units_sold"),
            )
            .join(Product, OrderItem.product_id == Product.product_id)
            .filter(Product.status == "Active")
            .group_by(Product.product_id)
            .order_by(func.sum(OrderItem.subtotal).desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "product_id": row.product_id,
                "product_name": row.product_name,
                "brand": row.brand,
                "category": row.category,
                "final_price": float(row.final_price or 0),
                "revenue": float(row.revenue or 0),
                "units_sold": int(row.units_sold or 0),
            }
            for row in rows
        ]

    def highest_spending_customers(self, db: Session, limit: int = 10) -> list[dict]:
        """
        Return customers ranked by total amount spent.

        Args:
            db:    Active SQLAlchemy session.
            limit: Maximum rows to return.

        Returns:
            List of dicts with ``user_id``, ``total_spent``, ``order_count``.
        """
        logger.debug("OrderRepository.highest_spending_customers — limit=%d", limit)
        rows = (
            db.query(
                Order.user_id,
                func.round(func.sum(Order.total_amount), 2).label("total_spent"),
                func.count(Order.order_id).label("order_count"),
            )
            .group_by(Order.user_id)
            .order_by(func.sum(Order.total_amount).desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "user_id": row.user_id,
                "total_spent": float(row.total_spent or 0),
                "order_count": int(row.order_count or 0),
            }
            for row in rows
        ]

    def customer_order_stats(self, db: Session, user_id: str) -> dict:
        """
        Per-customer order statistics.

        Args:
            db:      Active SQLAlchemy session.
            user_id: Customer identifier.

        Returns:
            Dict with ``total_orders``, ``total_spent``, ``avg_order_value``.
        """
        logger.debug(
            "OrderRepository.customer_order_stats — user_id=%s", user_id
        )
        total_orders: int = (
            db.query(func.count(Order.order_id))
            .filter(Order.user_id == user_id)
            .scalar()
        )
        total_spent = (
            db.query(func.coalesce(func.sum(Order.total_amount), 0))
            .filter(Order.user_id == user_id)
            .scalar()
        )
        avg_value = (
            db.query(func.coalesce(func.avg(Order.total_amount), 0))
            .filter(Order.user_id == user_id)
            .scalar()
        )
        return {
            "total_orders": total_orders,
            "total_spent": round(float(total_spent), 2),
            "avg_order_value": round(float(avg_value), 2),
        }
