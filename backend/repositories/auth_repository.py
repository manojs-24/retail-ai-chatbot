

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.models import User

logger = get_logger(__name__)


class AuthRepository:


    def get_user_by_email(self, db: Session, email: str) -> User | None:

        logger.debug("Looking up user by email: %s", email)

        user: User | None = (
            db.query(User)
            .filter(User.email == email.lower().strip())
            .first()
        )

        if user:
            logger.debug("User found — user_id=%s role=%s", user.user_id, user.role)
        else:
            logger.debug("No user found for email: %s", email)

        return user
