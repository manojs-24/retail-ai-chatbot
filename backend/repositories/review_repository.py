"""
Review Repository
=================
Pure data-access layer for the ``product_reviews`` table.

No business logic lives here.
"""

from __future__ import annotations

import logging
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.models import ProductReview

logger = get_logger(__name__)


class ReviewRepository:
    """Data-access layer for :class:`~backend.models.models.ProductReview` records."""

    def get_by_product(
        self,
        db: Session,
        product_id: str,
        limit: int = 5,
    ) -> list[ProductReview]:
        """
        Return the most recent reviews for a product, newest first.

        Args:
            db:         Active SQLAlchemy session.
            product_id: Product identifier.
            limit:      Maximum number of reviews to return.

        Returns:
            List of :class:`~backend.models.models.ProductReview` objects.
        """
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
        """
        Compute aggregate review statistics for a single product.

        Args:
            db:         Active SQLAlchemy session.
            product_id: Product identifier.

        Returns:
            Dict with keys ``avg_rating`` and ``review_count``.
        """
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
        """
        Return all reviews written by a specific user.

        Args:
            db:      Active SQLAlchemy session.
            user_id: Customer identifier.

        Returns:
            List of :class:`~backend.models.models.ProductReview` objects.
        """
        logger.debug("ReviewRepository.get_by_user — user_id=%s", user_id)
        return (
            db.query(ProductReview)
            .filter(ProductReview.user_id == user_id)
            .order_by(ProductReview.review_date.desc())
            .all()
        )
