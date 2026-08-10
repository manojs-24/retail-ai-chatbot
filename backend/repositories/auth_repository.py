"""
Authentication Repository
=========================
Responsible for ONE thing only: querying the database for a user record.

No business logic lives here — this layer simply translates a Python call
into a SQL query and returns a raw SQLAlchemy model instance (or None).

Used by:
    backend.services.auth_service
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.models import User

logger = get_logger(__name__)


class AuthRepository:
    """
    Data-access layer for authentication-related queries.

    Each method receives an active :class:`~sqlalchemy.orm.Session` so the
    caller (the service layer) controls the session lifecycle.
    """

    def get_user_by_email(self, db: Session, email: str) -> User | None:
        """
        Fetch a user row by email address.

        Args:
            db:    An active SQLAlchemy database session.
            email: The email address to look up (case-insensitive).

        Returns:
            The :class:`~backend.models.models.User` ORM object if found,
            otherwise ``None``.
        """
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
