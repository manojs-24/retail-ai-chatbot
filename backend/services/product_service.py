from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.repositories.product_repository import ProductRepository
from backend.repositories.review_repository import ReviewRepository

logger = get_logger(__name__)


class ProductService:
    """Service layer for product data operations."""

    def __init__(self) -> None:
        self._product_repo = ProductRepository()
        self._review_repo = ReviewRepository()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def search_products(
        self, db: Session, keyword: str, limit: int = 10
    ) -> dict:

        logger.info(
            "ProductService.search_products — keyword=%r limit=%d", keyword, limit
        )
        products = self._product_repo.search(db, keyword, limit)

        results = []
        for p in products:
            stats = self._review_repo.get_stats_by_product(db, p.product_id)
            results.append(
                {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "brand": p.brand,
                    "category": p.category,
                    "sub_category": p.sub_category,
                    "description": p.description,
                    "price": p.price,
                    "discount_percentage": p.discount_percentage,
                    "final_price": p.final_price,
                    "stock_quantity": p.stock_quantity,
                    "warranty_months": p.warranty_months,
                    "color": p.color,
                    "specifications": p.specifications,
                    "avg_rating": stats["avg_rating"],
                    "review_count": stats["review_count"],
                }
            )

        return {"keyword": keyword, "count": len(results), "products": results}

    def get_product_details(self, db: Session, product_id: str) -> dict | None:

        logger.info(
            "ProductService.get_product_details — product_id=%s", product_id
        )
        product = self._product_repo.get_by_id(db, product_id)
        if product is None:
            return None

        stats = self._review_repo.get_stats_by_product(db, product_id)
        return {
            "product_id": product.product_id,
            "product_name": product.product_name,
            "brand": product.brand,
            "category": product.category,
            "sub_category": product.sub_category,
            "description": product.description,
            "sku": product.sku,
            "price": product.price,
            "discount_percentage": product.discount_percentage,
            "final_price": product.final_price,
            "stock_quantity": product.stock_quantity,
            "reorder_level": product.reorder_level,
            "warranty_months": product.warranty_months,
            "weight": product.weight,
            "color": product.color,
            "specifications": product.specifications,
            "supplier_name": product.supplier_name,
            "launch_date": str(product.launch_date) if product.launch_date else None,
            "status": product.status,
            "avg_rating": stats["avg_rating"],
            "review_count": stats["review_count"],
        }

    def get_product_reviews(
        self, db: Session, product_id: str, limit: int = 10
    ) -> dict:

        logger.info(
            "ProductService.get_product_reviews — product_id=%s limit=%d",
            product_id, limit,
        )
        reviews = self._review_repo.get_by_product(db, product_id, limit=limit)
        stats = self._review_repo.get_stats_by_product(db, product_id)

        return {
            "product_id": product_id,
            "avg_rating": stats["avg_rating"],
            "review_count": stats["review_count"],
            "count": len(reviews),
            "reviews": [
                {
                    "review_id": r.review_id,
                    "user_id": r.user_id,
                    "rating": r.rating,
                    "review_text": r.review_text,
                    "sentiment": r.sentiment,
                    "review_date": str(r.review_date) if r.review_date else None,
                }
                for r in reviews
            ],
        }
