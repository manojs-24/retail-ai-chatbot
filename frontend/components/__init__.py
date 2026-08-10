"""
frontend.components — Reusable Streamlit UI components.

Each function in this package renders a self-contained UI fragment.
Components should accept plain Python values as arguments and must not
maintain their own ``st.session_state`` keys — state ownership belongs
to the page that calls the component.
"""
