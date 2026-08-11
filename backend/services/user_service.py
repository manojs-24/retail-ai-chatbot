

from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.repositories.user_repository import UserRepository

logger = get_logger(__name__)


class UserService:
    """Service layer for user data operations."""

    def __init__(self) -> None:
        self._repo = UserRepository()

    def get_customer_profile(self, db: Session, user_id: str) -> dict | None:

        logger.info("UserService.get_customer_profile — user_id=%s", user_id)
        user = self._repo.get_by_id(db, user_id)
        if user is None:
            logger.warning("User not found — user_id=%s", user_id)
            return None

        return {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "phone": user.phone,
            "gender": user.gender,
            "age": user.age,
            "city": user.city,
            "state": user.state,
            "join_date": str(user.join_date) if user.join_date else None,
            "total_orders": user.total_orders,
            "total_spent": user.total_spent,
            "preferred_category": user.preferred_category,
            "loyalty_level": user.loyalty_level,
            "role": user.role,
        }
