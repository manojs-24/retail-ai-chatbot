
from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.repositories.order_repository import OrderRepository

logger = get_logger(__name__)


def _serialise_order(order) -> dict:
    """Convert an Order ORM object to a plain dict."""
    return {
        "order_id": order.order_id,
        "order_date": str(order.order_date) if order.order_date else None,
        "total_amount": order.total_amount,
        "payment_method": order.payment_method,
        "payment_status": order.payment_status,
        "delivery_status": order.delivery_status,
        "delivery_date": str(order.delivery_date) if order.delivery_date else None,
        "shipping_address": order.shipping_address,
    }


def _serialise_item(item, product) -> dict:
    """Convert an (OrderItem, Product) tuple to a plain dict."""
    return {
        "order_item_id": item.order_item_id,
        "product_id": product.product_id,
        "product_name": product.product_name,
        "brand": product.brand,
        "category": product.category,
        "quantity": item.quantity,
        "unit_price": item.unit_price,
        "discount": item.discount,
        "subtotal": item.subtotal,
    }


class OrderService:
    """Service layer for order data operations."""

    def __init__(self) -> None:
        self._repo = OrderRepository()

    # ------------------------------------------------------------------
    # Customer-facing — always scoped to user_id
    # ------------------------------------------------------------------

    def get_purchase_history(self, db: Session, user_id: str) -> dict:

        logger.info("OrderService.get_purchase_history — user_id=%s", user_id)
        orders = self._repo.get_orders_by_user(db, user_id)
        return {
            "user_id": user_id,
            "total_orders": len(orders),
            "orders": [_serialise_order(o) for o in orders],
        }

    def get_recent_orders(self, db: Session, user_id: str, limit: int = 5) -> dict:

        logger.info(
            "OrderService.get_recent_orders — user_id=%s limit=%d", user_id, limit
        )
        orders = self._repo.get_orders_by_user(db, user_id, limit=limit)
        return {
            "user_id": user_id,
            "count": len(orders),
            "orders": [_serialise_order(o) for o in orders],
        }

    def get_order_details(
        self, db: Session, order_id: str, user_id: str
    ) -> dict | None:

        logger.info(
            "OrderService.get_order_details — order_id=%s user_id=%s",
            order_id, user_id,
        )
        order = self._repo.get_order_by_id(db, order_id)
        if order is None:
            logger.warning("Order not found — order_id=%s", order_id)
            return None

        # Ownership check — customer can NEVER access another customer's order.
        if order.user_id != user_id:
            logger.warning(
                "Access denied — order_id=%s belongs to user_id=%s, requested by %s",
                order_id, order.user_id, user_id,
            )
            return None

        item_rows = self._repo.get_items_with_products(db, order_id)
        return {
            **_serialise_order(order),
            "items": [_serialise_item(item, product) for item, product in item_rows],
        }

    # ------------------------------------------------------------------
    # Manager-facing — unrestricted by user_id
    # ------------------------------------------------------------------

    def get_order_details_for_manager(
        self, db: Session, order_id: str
    ) -> dict | None:

        logger.info(
            "OrderService.get_order_details_for_manager — order_id=%s", order_id
        )
        return self._repo.get_order_details_for_manager(db, order_id)

    def get_customer_orders_for_manager(
        self, db: Session, user_id: str
    ) -> dict:

        logger.info(
            "OrderService.get_customer_orders_for_manager — user_id=%s", user_id
        )
        orders = self._repo.get_orders_by_user(db, user_id)
        stats = self._repo.customer_order_stats(db, user_id)
        return {
            "user_id": user_id,
            "total_orders": len(orders),
            "total_spent": stats["total_spent"],
            "avg_order_value": stats["avg_order_value"],
            "orders": [_serialise_order(o) for o in orders],
        }

    def get_sales_summary(self, db: Session) -> dict:

        logger.info("OrderService.get_sales_summary")
        return self._repo.sales_summary(db)

    def get_monthly_sales(self, db: Session) -> dict:

        logger.info("OrderService.get_monthly_sales")
        months = self._repo.monthly_sales(db)
        return {"months": months, "total_months": len(months)}
