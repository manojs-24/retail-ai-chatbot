"""
Functions
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

# Module-level service singletons (stateless — safe to reuse)
_order_svc = OrderService()
_product_svc = ProductService()
_user_svc = UserService()


def get_purchase_history(user_id: str) -> dict:

    logger.info("customer_sql_tool.get_purchase_history — user_id=%s", user_id)
    db = SessionLocal()
    try:
        return _order_svc.get_purchase_history(db, user_id)
    finally:
        db.close()


def get_recent_orders(user_id: str, limit: int = 5) -> dict:

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

    logger.info("customer_sql_tool.search_products — keyword=%r", keyword)
    db = SessionLocal()
    try:
        return _product_svc.search_products(db, keyword)
    finally:
        db.close()


def get_product_details(product_id: str) -> dict | None:

    logger.info(
        "customer_sql_tool.get_product_details — product_id=%s", product_id
    )
    db = SessionLocal()
    try:
        return _product_svc.get_product_details(db, product_id)
    finally:
        db.close()


def get_product_reviews(product_id: str, limit: int = 10) -> dict:

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

    logger.info(
        "customer_sql_tool.recommend_products — user_id=%s limit=%d",
        user_id, limit,
    )
    db = SessionLocal()
    try:
        return recommend_for_customer(db, user_id, limit=limit)
    finally:
        db.close()
