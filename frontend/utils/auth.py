"""
Frontend Authentication Utilities
===================================
Reusable helpers for:
    - Calling the backend login/logout API.
    - Reading and writing session state.
    - Protecting pages (guard functions).

Import these functions into any Streamlit page — never duplicate this logic.
"""

from __future__ import annotations

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Backend base URL — change this if your backend runs on a different port
# ---------------------------------------------------------------------------
BACKEND_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_login_api(email: str, password: str) -> dict | None:
    """
    POST /auth/login → returns the parsed JSON dict on success, or None on failure.

    Args:
        email:    The user's email address.
        password: The user's plain-text password.

    Returns:
        A dict with keys ``user_id``, ``full_name``, ``email``, ``role``
        if credentials are valid, or ``None`` otherwise.
    """
    try:
        response = httpx.post(
            f"{BACKEND_URL}/auth/login",
            json={"email": email, "password": password},
            timeout=10.0,
        )
        if response.status_code == 200:
            return response.json()
        return None
    except httpx.RequestError:
        st.error("⚠️ Cannot connect to the backend. Make sure it is running.")
        return None


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def set_session(user: dict) -> None:
    """
    Write user data returned by the API into ``st.session_state``.

    Keys set:
        - ``logged_in``  (bool)
        - ``user_id``    (str)
        - ``full_name``  (str)
        - ``email``      (str)
        - ``role``       (str)

    Args:
        user: The dict returned by :func:`call_login_api`.
    """
    st.session_state["logged_in"] = True
    st.session_state["user_id"] = user["user_id"]
    st.session_state["full_name"] = user["full_name"]
    st.session_state["email"] = user["email"]
    st.session_state["role"] = user["role"]


def clear_session() -> None:
    """
    Remove all authentication keys from ``st.session_state``.
    Called on logout.
    """
    for key in ["logged_in", "user_id", "full_name", "email", "role"]:
        st.session_state.pop(key, None)


def is_logged_in() -> bool:
    """Return True if a valid session exists."""
    return bool(st.session_state.get("logged_in", False))


def get_role() -> str:
    """Return the current user's role, or an empty string if not logged in."""
    return st.session_state.get("role", "")


# ---------------------------------------------------------------------------
# Page guards
# ---------------------------------------------------------------------------

def require_login() -> None:
    """
    Redirect to the Login page if the user is not authenticated.
    Call this at the top of every protected page.
    """
    if not is_logged_in():
        st.switch_page("pages/Login.py")


def require_role(expected_role: str) -> None:
    """
    Show an 'Access Denied' message if the logged-in user's role does not match.
    Call after :func:`require_login`.

    Args:
        expected_role: ``"customer"`` or ``"manager"``.
    """
    require_login()
    if get_role() != expected_role:
        st.error("🚫 Access Denied — you do not have permission to view this page.")
        st.stop()
