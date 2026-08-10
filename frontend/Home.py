"""
Streamlit frontend entry point — Home / Landing page.

Run locally with::

    uv run streamlit run frontend/Home.py --server.port 8501
"""

from __future__ import annotations

import streamlit as st

from frontend.utils.auth import is_logged_in, get_role

# ---------------------------------------------------------------------------
# Page configuration (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail AI System",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# If already logged in, redirect straight to the correct dashboard
# ---------------------------------------------------------------------------
if is_logged_in():
    if get_role() == "manager":
        st.switch_page("pages/ManagerDashboard.py")
    else:
        st.switch_page("pages/CustomerDashboard.py")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🛒 Retail AI System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:grey; font-size:1.1rem;'>"
    "AI-Powered Smart Retail Intelligence &amp; Recommendation System"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Feature highlights
# ---------------------------------------------------------------------------
st.markdown("### 🚀 What You Can Do")

st.info(
    """
    **AI-Powered Retail Intelligence Platform** that helps customers discover products, 
    track orders, and get instant support, while enabling store managers to monitor 
    inventory, analyze sales, gain customer insights, and make data-driven decisions 
    using AI-powered recommendations, analytics, and intelligent chat assistance.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Login button — navigates to the Login page
# ---------------------------------------------------------------------------
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    if st.button("🔐 Login", width='stretch', type="primary"):
        st.switch_page("pages/Login.py")

st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "Retail AI System · v1.0.0 · Powered by LangChain &amp; FastAPI"
    "</p>",
    unsafe_allow_html=True,
)
