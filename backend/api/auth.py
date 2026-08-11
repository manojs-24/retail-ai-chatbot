from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.logging import get_logger
from backend.schemas.auth import LoginRequest, LoginResponse
from backend.services.auth_service import AuthService

logger = get_logger(__name__)

# All routes in this file are prefixed with /auth
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Service instance — stateless, safe to reuse across requests
_auth_service = AuthService()


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login with email and password",
    status_code=status.HTTP_200_OK,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
) -> LoginResponse:

    result = _auth_service.login(db, request)

    if result is None:
        logger.warning("Returning 401 for failed login: %s", request.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password. Please try again.",
        )

    return result


@router.post(
    "/logout",
    summary="Logout (server acknowledgement)",
    status_code=status.HTTP_200_OK,
)
def logout() -> dict[str, str]:
    return {"message": "Logged out successfully."}
