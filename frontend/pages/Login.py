from __future__ import annotations

import streamlit as st

from frontend.utils.auth import call_login_api, hide_streamlit_nav, is_logged_in, set_session


# Page config

st.set_page_config(
    page_title="Login — Retail AI",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Hide the sidebar and Streamlit's auto-generated nav links (user not yet logged in)
hide_streamlit_nav()
st.markdown(
    "<style>[data-testid='stSidebar']{display:none}</style>",
    unsafe_allow_html=True,
)


# If already logged in, redirect immediately (no flash of login form)

if is_logged_in():
    role = st.session_state.get("role", "")
    if role == "manager":
        st.switch_page("pages/ManagerDashboard.py")
    else:
        st.switch_page("pages/CustomerDashboard.py")


# Page header

st.markdown(
    "<h2 style='text-align:center;'>🔐 Login</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:grey;'>Sign in to your Retail AI account</p>",
    unsafe_allow_html=True,
)
st.divider()


# Login form

col_l, col_c, col_r = st.columns([1, 2, 1])

with col_c:
    with st.form(key="login_form", clear_on_submit=False):
        email = st.text_input(
            "📧 Email",
            placeholder="you@example.com",
            autocomplete="email",
        )
        password = st.text_input(
            "🔑 Password",
            type="password",
            placeholder="Enter your password",
            autocomplete="current-password",
        )

        submitted = st.form_submit_button(
            "Login →",
            width='stretch',
            type="primary",
        )

    
    # Handle form submission
    
    if submitted:
        if not email or not password:
            st.warning("⚠️ Please fill in both email and password.")
        else:
            with st.spinner("Verifying credentials..."):
                user = call_login_api(email.strip(), password)

            if user:
                set_session(user)
                st.success(f"✅ Welcome back, {user['full_name']}!")

                # Route based on role
                if user["role"] == "manager":
                    st.switch_page("pages/ManagerDashboard.py")
                else:
                    st.switch_page("pages/CustomerDashboard.py")
            else:
                st.error("❌ Invalid email or password. Please try again.")

    
    # Back link
    
    st.markdown("---")
    if st.button("← Back to Home", width='stretch'):
        st.switch_page("Home.py")
