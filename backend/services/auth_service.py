
from __future__ import annotations

from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.repositories.auth_repository import AuthRepository
from backend.schemas.auth import LoginRequest, LoginResponse

logger = get_logger(__name__)


class AuthService:

    def __init__(self) -> None:
        self._repo = AuthRepository()

    def login(self, db: Session, request: LoginRequest) -> LoginResponse | None:

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
