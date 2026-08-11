"""
Customer Chatbot Page
=====================
Streamlit page that renders the customer-facing AI chatbot with multilingual
voice support powered by Sarvam AI.

Voice flow:
  1. User records audio via 🎤 st.audio_input.
  2. Sarvam STT converts audio → transcript + detected language_code.
  3. Transcript feeds the existing LangGraph pipeline unchanged.
  4. Response is translated into the detected language via Sarvam Translate.
  5. 🔊 Speak Response converts the final text to audio via Sarvam TTS.

Typed flow:
  1. User types a query.
  2. Sarvam Translate detect-API identifies the input language.
  3. LangGraph pipeline runs unchanged.
  4. Response is translated into the detected language.
  5. 🔊 Speak Response works the same way.

Nothing in the LangGraph / RAG / SQL / ML / memory pipeline is modified.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import streamlit as st

from frontend.utils.auth import clear_session, require_role
from frontend.utils.chat_memory import SessionMemory
from frontend.utils.sarvam import (
    LANGUAGE_NAMES,
    stt_from_audio,
    translate_response,
    translate_to_english,
    tts_to_audio,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Chatbot — Retail AI",
    page_icon="💬",
    layout="wide",
)

# Guard — only "customer" role may proceed
require_role("customer")

# Lazy import of the graph (after path + dotenv are set up)
from backend.graph.customer_graph import customer_graph  # noqa: E402

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "customer_chat_memory" not in st.session_state:
    st.session_state["customer_chat_memory"] = SessionMemory()

if "customer_chat_history" not in st.session_state:
    st.session_state["customer_chat_history"] = []

# Current-query language code (reset each query, no persistent memory)
if "current_lang" not in st.session_state:
    st.session_state["current_lang"] = "en-IN"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state['full_name']}")
    st.markdown(f"📧 {st.session_state['email']}")
    st.markdown(f"🏷️ Role: `{st.session_state['role']}`")
    st.divider()
    if st.button("🏠 Dashboard", width="stretch"):
        st.switch_page("pages/CustomerDashboard.py")
    if st.button("🚪 Logout", width="stretch", type="secondary"):
        clear_session()
        st.switch_page("Home.py")
    st.divider()
    st.caption("💡 Powered by LangChain + LangGraph")

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("💬 Customer AI Assistant")
st.markdown(
    "Ask me anything about our **policies**, **products**, **orders**, "
    "or get **personalised recommendations**."
)

if st.button("🗑️ Clear Chat"):
    st.session_state["customer_chat_history"] = []
    st.session_state["customer_chat_memory"] = SessionMemory()
    st.session_state["current_lang"] = "en-IN"
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
        if st.button(question, width="stretch", key=f"suggest_{question}"):
            st.session_state["pending_input"] = question
            st.session_state["current_lang"]  = "en-IN"   # pill questions are English

st.divider()

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------
for message in st.session_state["customer_chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "meta" in message:
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
# Voice input  (🎤) — sits just above the chat input bar
# ---------------------------------------------------------------------------
audio = st.audio_input("🎤 Record your message", key="voice_input_customer")

if audio:
    # st.audio_input keeps the same object in widget state across reruns.
    # Use the audio's byte content as a fingerprint so we only transcribe
    # a new recording once, not on every subsequent rerun.
    audio_bytes_raw = audio.read()
    audio_id = hash(audio_bytes_raw)
    audio.seek(0)  # rewind so stt_from_audio can read it again

    if audio_id != st.session_state.get("_last_audio_id"):
        st.session_state["_last_audio_id"] = audio_id
        with st.spinner("Transcribing audio…"):
            transcript, lang_code = stt_from_audio(audio)

        if transcript:
            # Translate the regional transcript to English for the backend
            if lang_code != "en-IN":
                with st.spinner("Translating voice input to English…"):
                    english_transcript, _ = translate_to_english(transcript)
            else:
                english_transcript = transcript
            st.success(
                f"✅ Transcribed: *{transcript}* "
                f"— Language: **{LANGUAGE_NAMES.get(lang_code, lang_code)}**"
            )
            # Store original text for display, English text for the graph
            st.session_state["pending_input"]         = english_transcript
            st.session_state["pending_display_text"]  = transcript
            st.session_state["current_lang"]          = lang_code
        else:
            st.warning("⚠️ Could not transcribe audio. Please type your question below.")

# ---------------------------------------------------------------------------
# Consume pending input (voice or pill)
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_input", None)

# ---------------------------------------------------------------------------
# Chat text input
# ---------------------------------------------------------------------------
user_input: str | None = st.chat_input("Ask me anything about our store…")

# Prefer voice/pill input, then typed input
query = pending or user_input

if query:
    # For typed input: translate to English + detect language in one call.
    # For voice/pill input: pending already holds the English translation;
    # current_lang was set when the audio was processed.
    if query == user_input and user_input:
        try:
            with st.spinner("Detecting language & translating…"):
                english_query, detected = translate_to_english(query)
            st.session_state["current_lang"] = detected
            # Debug box — remove once confirmed working
            st.info(
                f"🔍 Debug | original: `{query[:80]}` | "
                f"→ english: `{english_query[:80]}` | "
                f"lang: `{detected}`"
            )
        except EnvironmentError as exc:
            st.error(f"⚠️ Sarvam API key not configured: {exc}")
            # Fall back: treat input as English so the pipeline still works
            english_query = query
            st.session_state["current_lang"] = "en-IN"
        except Exception as exc:  # noqa: BLE001
            st.warning(f"⚠️ Language detection failed ({exc}) — treating input as English.")
            english_query = query
            st.session_state["current_lang"] = "en-IN"
    else:
        # Voice path: query is already the English translation of the transcript.
        # The original display text was saved separately.
        english_query = query

    # What to show the user — their original words, not the translation
    display_query = st.session_state.pop("pending_display_text", None) or query

    lang_code = st.session_state["current_lang"]
    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)

    if lang_code != "en-IN":
        st.caption(f"🌐 Detected language: **{lang_name}** — responding in {lang_name}")

    # Display the original (regional) text to the user
    with st.chat_message("user"):
        st.markdown(display_query)
    st.session_state["customer_chat_history"].append(
        {"role": "user", "content": display_query}
    )

    # Load conversation context (unchanged pipeline)
    memory: SessionMemory = st.session_state["customer_chat_memory"]
    conversation_context = memory.load_context()

    logger.debug(
        "Customer query=%r | english=%r | lang=%s | memory_turns=%d",
        display_query[:80], english_query[:80], lang_code, memory.turn_count,
    )

    # Invoke the LangGraph with the ENGLISH query (backend expects English)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            graph_input = {
                "query":                english_query,
                "user_id":              st.session_state.get("user_id", ""),
                "role":                 "customer",
                "conversation_context": conversation_context,
            }
            result: dict = customer_graph.invoke(graph_input)

        response: str = result.get("response", "I'm sorry, I couldn't generate a response.")
        intent: str   = result.get("intent", "N/A")
        raw_docs: list = result.get("retrieved_documents", [])

        # Translate response if input was not English
        if lang_code != "en-IN":
            with st.spinner(f"Translating response to {lang_name}…"):
                response = translate_response(response, lang_code)

        logger.debug("Customer response — intent=%s | lang=%s | len=%d", intent, lang_code, len(response))

        st.markdown(response)

        # Details expander
        sources = [
            {
                "document_name": doc["metadata"].get("document_name", "Unknown"),
                "page":          doc["metadata"].get("page", "?"),
            }
            for doc in raw_docs
        ]
        with st.expander("🔍 Details", expanded=False):
            st.markdown(f"**Detected Intent:** `{intent}`")
            if sources:
                st.markdown("**Retrieved Sources:**")
                for src in sources:
                    st.markdown(f"- 📄 `{src['document_name']}` — page {src['page']}")

    # Save to memory (unchanged)
    memory.save_context(query, response)

    # Persist to chat history — store lang so Speak works on past messages too
    msg_id = len(st.session_state["customer_chat_history"])
    st.session_state["customer_chat_history"].append(
        {
            "role":    "assistant",
            "content": response,
            "lang":    lang_code,
            "msg_id":  msg_id,
            "meta":    {"intent": intent, "sources": sources},
        }
    )
