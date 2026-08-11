from __future__ import annotations

import logging
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.models import ProductReview

logger = get_logger(__name__)


class ReviewRepository:

    def get_by_product(
        self,
        db: Session,
        product_id: str,
        limit: int = 5,
    ) -> list[ProductReview]:

        logger.debug(
            "ReviewRepository.get_by_product — product_id=%s limit=%d",
            product_id, limit,
        )
        return (
            db.query(ProductReview)
            .filter(ProductReview.product_id == product_id)
            .order_by(ProductReview.review_date.desc())
            .limit(limit)
            .all()
        )

    def get_stats_by_product(self, db: Session, product_id: str) -> dict:

        logger.debug(
            "ReviewRepository.get_stats_by_product — product_id=%s", product_id
        )
        row = (
            db.query(
                func.round(func.avg(ProductReview.rating), 2).label("avg_rating"),
                func.count(ProductReview.review_id).label("review_count"),
            )
            .filter(ProductReview.product_id == product_id)
            .one()
        )
        return {
            "avg_rating": float(row.avg_rating or 0),
            "review_count": int(row.review_count or 0),
        }

    def get_by_user(self, db: Session, user_id: str) -> list[ProductReview]:

        logger.debug("ReviewRepository.get_by_user — user_id=%s", user_id)
        return (
            db.query(ProductReview)
            .filter(ProductReview.user_id == user_id)
            .order_by(ProductReview.review_date.desc())
            .all()
        )
