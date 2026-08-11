from __future__ import annotations

import httpx
import streamlit as st


BACKEND_URL = "http://127.0.0.1:8000"


# API calls

def call_login_api(email: str, password: str) -> dict | None:
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


def set_session(user: dict) -> None:

    st.session_state["logged_in"] = True
    st.session_state["user_id"] = user["user_id"]
    st.session_state["full_name"] = user["full_name"]
    st.session_state["email"] = user["email"]
    st.session_state["role"] = user["role"]


def clear_session() -> None:
    for key in ["logged_in", "user_id", "full_name", "email", "role"]:
        st.session_state.pop(key, None)


def is_logged_in() -> bool:
    return bool(st.session_state.get("logged_in", False))


def get_role() -> str:
    return st.session_state.get("role", "")







def hide_streamlit_nav() -> None:
    st.markdown(
        "<style>[data-testid='stSidebarNav']{display:none}</style>",
        unsafe_allow_html=True,
    )


def require_login() -> None:
    hide_streamlit_nav()
    if not is_logged_in():
        st.switch_page("pages/Login.py")


def require_role(expected_role: str) -> None:
    hide_streamlit_nav()
    require_login()
    if get_role() != expected_role:
        st.error("🚫 Access Denied — you do not have permission to view this page.")
        st.stop()
