"""
Authentication Schemas
======================
Pydantic models for login request and response payloads.
These are the data contracts between the API layer and the outside world.
"""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """
    Payload sent by the client when logging in.

    Fields:
        email    — the user's registered email address.
        password — plain-text password (no hashing in this lightweight implementation).
    """

    email: EmailStr = Field(..., description="User's registered email address")
    password: str = Field(..., min_length=1, description="User's plain-text password")


class LoginResponse(BaseModel):
    """
    Payload returned on successful login.

    Fields:
        user_id   — unique user identifier (used to personalise RAG / SQL queries).
        full_name — display name shown in the UI.
        email     — echoed back so the frontend can store it in session state.
        role      — either ``"customer"`` or ``"manager"``, drives page routing.
    """

    user_id: str = Field(..., description="Unique user identifier")
    full_name: str = Field(..., description="User's display name")
    email: EmailStr = Field(..., description="User's email address")
    role: str = Field(..., description="User role: 'customer' or 'manager'")
