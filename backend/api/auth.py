"""
Authentication API Router
=========================
Exposes HTTP endpoints related to user authentication.

Endpoints:
    POST /auth/login   — validate credentials, return user info.
    POST /auth/logout  — client-side logout (clears session state on the frontend).

This layer handles only HTTP concerns:
    - Parsing the request body into a Pydantic schema.
    - Calling the service layer.
    - Mapping results to appropriate HTTP status codes.
"""

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
    """
    Authenticate a user with their email and password.

    - Looks up the user in the **Users** table.
    - Compares the provided password directly (no hashing).
    - Returns ``user_id``, ``full_name``, ``email``, and ``role`` on success.
    - Returns **401 Unauthorized** if credentials are invalid.

    The frontend stores these fields in ``st.session_state`` and uses
    ``role`` to decide which dashboard to show.
    """
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
    """
    Acknowledge a logout request.

    No server-side session exists to invalidate (this is a stateless system).
    The actual logout is performed on the Streamlit frontend by clearing
    ``st.session_state``.

    This endpoint exists so the frontend has a consistent API surface
    and for future extensibility (e.g. token revocation).
    """
    return {"message": "Logged out successfully."}
