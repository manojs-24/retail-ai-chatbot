from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.models import User

logger = get_logger(__name__)


class UserRepository:
    """Data-access layer for :class:`~backend.models.models.User` records."""

    def get_by_id(self, db: Session, user_id: str) -> User | None:

        logger.debug("UserRepository.get_by_id — user_id=%s", user_id)
        return db.query(User).filter(User.user_id == user_id).first()

    def get_by_email(self, db: Session, email: str) -> User | None:

        logger.debug("UserRepository.get_by_email — email=%s", email)
        return (
            db.query(User)
            .filter(User.email == email.lower().strip())
            .first()
        )

    def get_all_customers(self, db: Session) -> list[User]:

        logger.debug("UserRepository.get_all_customers")
        return (
            db.query(User)
            .filter(User.role == "customer")
            .order_by(User.full_name)
            .all()
        )
