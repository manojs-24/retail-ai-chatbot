"""
Authentication Service
======================
Contains the business logic for authenticating a user.

This layer sits between the API router and the repository.
It decides *what* to do with the data returned from the DB.

Principle:
    - Repository  → pure DB I/O
    - Service     → business rules (credential check, error handling)
    - API router  → HTTP concerns only (status codes, request/response parsing)
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.repositories.auth_repository import AuthRepository
from backend.schemas.auth import LoginRequest, LoginResponse

logger = get_logger(__name__)


class AuthService:
    """
    Handles the authentication workflow.

    Depends on:
        :class:`~backend.repositories.auth_repository.AuthRepository`
    """

    def __init__(self) -> None:
        self._repo = AuthRepository()

    def login(self, db: Session, request: LoginRequest) -> LoginResponse | None:
        """
        Validate credentials and return a LoginResponse if they match.

        Steps:
            1. Fetch the user row by email from the DB (via repository).
            2. Compare the stored password directly against the provided password.
               (Plain-text comparison — intentional for this lightweight system.)
            3. Return a :class:`~backend.schemas.auth.LoginResponse` on success,
               or ``None`` if the email does not exist or the password is wrong.

        Args:
            db:      Active SQLAlchemy session (injected by FastAPI dependency).
            request: The validated login payload.

        Returns:
            A :class:`~backend.schemas.auth.LoginResponse` on success,
            or ``None`` on failure (wrong email / password).
        """
        logger.info("Login attempt for email: %s", request.email)

        # Step 1 — look up the user by email
        user = self._repo.get_user_by_email(db, request.email)

        if user is None:
            logger.warning("Login failed — email not found: %s", request.email)
            return None

        # Step 2 — direct plain-text password comparison
        if user.password != request.password:
            logger.warning("Login failed — wrong password for email: %s", request.email)
            return None

        # Step 3 — credentials matched, build the response
        logger.info(
            "Login successful — user_id=%s role=%s", user.user_id, user.role
        )

        return LoginResponse(
            user_id=str(user.user_id),
            full_name=str(user.full_name),
            email=str(user.email),
            role=str(user.role),
        )
