"""
Customer Chatbot Page
=====================
Streamlit page that renders the customer-facing AI chatbot.

Flow:
    1. Guard — requires "customer" role.
    2. Render the persistent sidebar (user info + logout).
    3. Render suggested questions as clickable pills.
    4. Render full chat history from ``st.session_state``.
    5. On user input → invoke the customer LangGraph with conversation context.
    6. Display detected intent, retrieved sources, and the AI response.

Memory:
    Uses ``SessionMemory`` (frontend/utils/chat_memory.py) stored in
    ``st.session_state["customer_chat_memory"]``.  Keeps recent turns verbatim
    and summarises older ones via the LLM, so the intent classifier can resolve
    pronoun/entity references (e.g. "its" → P0023) across turns.

Run with::

    uv run streamlit run frontend/Home.py --server.port 8501
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable when Streamlit runs the page directly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import streamlit as st

from frontend.utils.auth import clear_session, require_role
from frontend.utils.chat_memory import SessionMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Chatbot — Retail AI",
    page_icon="💬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Guard — only "customer" role may proceed
# ---------------------------------------------------------------------------
require_role("customer")

# ---------------------------------------------------------------------------
# Lazy import of the graph (after path + dotenv are set up)
# ---------------------------------------------------------------------------
from backend.graph.customer_graph import customer_graph  # noqa: E402

# ---------------------------------------------------------------------------
# Memory + chat history initialisation
# ---------------------------------------------------------------------------
if "customer_chat_memory" not in st.session_state:
    st.session_state["customer_chat_memory"] = SessionMemory()

if "customer_chat_history" not in st.session_state:
    st.session_state["customer_chat_history"] = []

# ---------------------------------------------------------------------------
# Sidebar — user info + navigation + logout
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state['full_name']}")
    st.markdown(f"📧 {st.session_state['email']}")
    st.markdown(f"🏷️ Role: `{st.session_state['role']}`")
    st.divider()
    if st.button("🏠 Dashboard", width='stretch'):
        st.switch_page("pages/CustomerDashboard.py")
    if st.button("🚪 Logout", width='stretch', type="secondary"):
        clear_session()
        st.switch_page("Home.py")
    st.divider()
    st.caption("💡 Powered by LangChain + LangGraph")

# ---------------------------------------------------------------------------
# Page header + Clear Chat button
# ---------------------------------------------------------------------------
st.title("💬 Customer AI Assistant")
st.markdown(
    "Ask me anything about our **policies**, **products**, **orders**, "
    "or get **personalised recommendations**."
)

if st.button("🗑️ Clear Chat"):
    st.session_state["customer_chat_history"] = []
    st.session_state["customer_chat_memory"] = SessionMemory()
    logger.info("Customer chat cleared.")
    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------
SUGGESTED_QUESTIONS = [
    "What is your return policy?",
    "What is the warranty on laptops?",
    "What are the shipping charges?",
    "Recommend me some Products",
    "Show my previous purchases",
]

st.markdown("**💡 Suggested questions:**")
cols = st.columns(len(SUGGESTED_QUESTIONS))
for col, question in zip(cols, SUGGESTED_QUESTIONS):
    with col:
        if st.button(question, width='stretch', key=f"suggest_{question}"):
            st.session_state["pending_input"] = question

st.divider()

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------
for message in st.session_state["customer_chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "meta" in message:
            meta = message["meta"]
            with st.expander("🔍 Details", expanded=False):
                st.markdown(f"**Detected Intent:** `{meta.get('intent', 'N/A')}`")
                sources = meta.get("sources", [])
                if sources:
                    st.markdown("**Retrieved Sources:**")
                    for src in sources:
                        st.markdown(
                            f"- 📄 `{src['document_name']}` — page {src['page']}"
                        )

# ---------------------------------------------------------------------------
# Consume pending input from a suggested-question button click
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_input", None)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input: str | None = st.chat_input("Ask me anything about our store…")

# Prefer pending pill input, then typed input.
query = pending or user_input

if query:
    # Display user message immediately.
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["customer_chat_history"].append(
        {"role": "user", "content": query}
    )

    # ------------------------------------------------------------------
    # Load conversation context from memory.
    # ------------------------------------------------------------------
    memory: SessionMemory = st.session_state["customer_chat_memory"]
    conversation_context = memory.load_context()

    logger.debug(
        "Customer query=%r | memory_turns=%d | context_len=%d",
        query[:80],
        memory.turn_count,
        len(conversation_context),
    )

    # ------------------------------------------------------------------
    # Invoke the customer LangGraph
    # ------------------------------------------------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            graph_input = {
                "query": query,
                "user_id": st.session_state.get("user_id", ""),
                "role": "customer",
                "conversation_context": conversation_context,
            }
            result: dict = customer_graph.invoke(graph_input)

        response: str = result.get("response", "I'm sorry, I couldn't generate a response.")
        intent: str = result.get("intent", "N/A")
        raw_docs: list = result.get("retrieved_documents", [])

        logger.debug("Customer response — intent=%s | response_len=%d", intent, len(response))

        st.markdown(response)

        # Summarise intent + sources in an expander.
        sources = [
            {
                "document_name": doc["metadata"].get("document_name", "Unknown"),
                "page": doc["metadata"].get("page", "?"),
            }
            for doc in raw_docs
        ]
        with st.expander("🔍 Details", expanded=False):
            st.markdown(f"**Detected Intent:** `{intent}`")
            if sources:
                st.markdown("**Retrieved Sources:**")
                for src in sources:
                    st.markdown(f"- 📄 `{src['document_name']}` — page {src['page']}")

    # ------------------------------------------------------------------
    # Save this turn to memory so the next query has context.
    # ------------------------------------------------------------------
    memory.save_context(query, response)

    # Persist assistant message + metadata to displayed history.
    st.session_state["customer_chat_history"].append(
        {
            "role": "assistant",
            "content": response,
            "meta": {"intent": intent, "sources": sources},
        }
    )
