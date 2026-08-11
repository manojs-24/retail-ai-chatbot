
from __future__ import annotations

import logging
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.models import OrderItem, Product

logger = get_logger(__name__)


class ProductRepository:
    """Data-access layer for :class:`~backend.models.models.Product` records."""

    def get_by_id(self, db: Session, product_id: str) -> Product | None:

        logger.debug("ProductRepository.get_by_id — product_id=%s", product_id)
        return db.query(Product).filter(Product.product_id == product_id).first()

    def search(
        self,
        db: Session,
        keyword: str,
        limit: int = 10,
    ) -> list[Product]:

        logger.debug("ProductRepository.search — keyword=%r limit=%d", keyword, limit)
        term = f"%{keyword.lower()}%"
        return (
            db.query(Product)
            .filter(
                Product.status == "Active",
                or_(
                    func.lower(Product.product_name).like(term),
                    func.lower(Product.brand).like(term),
                    func.lower(Product.category).like(term),
                    func.lower(Product.sub_category).like(term),
                ),
            )
            .order_by(Product.rating.desc())
            .limit(limit)
            .all()
        )

    def get_top_selling(self, db: Session, limit: int = 10) -> list[Product]:

        logger.debug("ProductRepository.get_top_selling — limit=%d", limit)
        return (
            db.query(Product)
            .filter(Product.status == "Active")
            .order_by(Product.total_sold.desc())
            .limit(limit)
            .all()
        )

    def get_low_stock(self, db: Session) -> list[Product]:

        logger.debug("ProductRepository.get_low_stock")
        return (
            db.query(Product)
            .filter(
                Product.status == "Active",
                Product.stock_quantity <= Product.reorder_level,
            )
            .order_by(Product.stock_quantity.asc())
            .all()
        )

    def get_out_of_stock(self, db: Session) -> list[Product]:

        logger.debug("ProductRepository.get_out_of_stock")
        return (
            db.query(Product)
            .filter(
                Product.status == "Active",
                Product.stock_quantity == 0,
            )
            .all()
        )

    def inventory_stats(self, db: Session) -> dict:

        logger.debug("ProductRepository.inventory_stats")
        active = db.query(Product).filter(Product.status == "Active")

        total_products: int = active.count()
        total_stock: int = (
            db.query(func.coalesce(func.sum(Product.stock_quantity), 0))
            .filter(Product.status == "Active")
            .scalar()
        )
        out_of_stock: int = active.filter(Product.stock_quantity == 0).count()
        low_stock: int = active.filter(
            Product.stock_quantity > 0,
            Product.stock_quantity <= Product.reorder_level,
        ).count()

        return {
            "total_products": total_products,
            "total_stock": int(total_stock),
            "out_of_stock": out_of_stock,
            "low_stock": low_stock,
        }

    def category_summary(self, db: Session) -> list[dict]:

        logger.debug("ProductRepository.category_summary")
        rows = (
            db.query(
                Product.category,
                func.count(Product.product_id).label("product_count"),
                func.sum(Product.stock_quantity).label("total_stock"),
                func.round(func.avg(Product.final_price), 2).label("avg_price"),
            )
            .filter(Product.status == "Active")
            .group_by(Product.category)
            .order_by(func.count(Product.product_id).desc())
            .all()
        )
        return [
            {
                "category": row.category,
                "product_count": row.product_count,
                "total_stock": int(row.total_stock or 0),
                "avg_price": float(row.avg_price or 0),
            }
            for row in rows
        ]

    def top_selling_with_revenue(
        self, db: Session, limit: int = 10
    ) -> list[dict]:

        logger.debug("ProductRepository.top_selling_with_revenue — limit=%d", limit)
        rows = (
            db.query(
                Product.product_id,
                Product.product_name,
                Product.brand,
                Product.category,
                Product.final_price,
                func.sum(OrderItem.quantity).label("units_sold"),
                func.round(func.sum(OrderItem.subtotal), 2).label("revenue"),
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
                "product_id": row.product_id,
                "product_name": row.product_name,
                "brand": row.brand,
                "category": row.category,
                "final_price": float(row.final_price or 0),
                "units_sold": int(row.units_sold or 0),
                "revenue": float(row.revenue or 0),
            }
            for row in rows
        ]
