"""
Session-level conversational summary memory.

Replaces the removed ``langchain.memory.ConversationSummaryBufferMemory`` with
an equivalent built on stable ``langchain_core`` primitives that ship with
LangChain 0.3.x.

Design
------
``SessionMemory`` stores turns as a list of ``(human, ai)`` string pairs plus an
optional running *summary* of older turns that have been compressed away.

When ``save_context`` is called and the total character count of the verbatim
buffer exceeds *max_chars* (default 3 000, roughly ~750 tokens), the oldest
half of the verbatim turns are summarised by the LLM and prepended to the
running summary.  The compressed turns are then dropped from the verbatim list.

``load_context`` returns a single plain string suitable for injecting into the
classifier / graph as ``conversation_context``.  Format::

    [Summary of earlier conversation]
    <summary text>

    Human: <turn n-k>
    AI: <turn n-k reply>
    ...
    Human: <turn n>
    AI: <turn n reply>

Usage
-----
::

    from frontend.utils.chat_memory import SessionMemory

    # initialise once per session
    if "customer_chat_memory" not in st.session_state:
        st.session_state["customer_chat_memory"] = SessionMemory()

    mem: SessionMemory = st.session_state["customer_chat_memory"]

    # before graph.invoke
    ctx = mem.load_context()

    # after graph.invoke
    mem.save_context(user_query, assistant_response)

    # clear chat
    st.session_state["customer_chat_memory"] = SessionMemory()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# How many characters of verbatim turns to keep before summarising.
# ~3 000 chars ≈ 750 tokens — comfortably under the 800-token budget we
# previously used with ConversationSummaryBufferMemory.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_CHARS = 3_000

# ---------------------------------------------------------------------------
# Prompt used to compress older turns into a running summary.
# ---------------------------------------------------------------------------
_SUMMARY_PROMPT = (
    "Progressively summarise the following conversation lines, adding to the "
    "summary provided.  Return only the updated summary — no commentary.\n\n"
    "Current summary:\n{summary}\n\n"
    "New conversation lines to add:\n{new_lines}\n\n"
    "Updated summary:"
)


@dataclass
class SessionMemory:
    """
    In-process, session-scoped conversational summary buffer.

    Parameters
    ----------
    max_chars:
        Maximum total characters of verbatim turn text to keep.
        When exceeded, the oldest half is compressed into ``summary``.
    """

    max_chars: int = _DEFAULT_MAX_CHARS
    summary: str = ""
    turns: list[tuple[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_context(self, human: str, ai: str) -> None:
        """Append a completed turn and compress if the buffer is too large."""
        self.turns.append((human, ai))
        self._maybe_compress()

    def load_context(self) -> str:
        """
        Return the full conversation context as a plain string.

        Returns an empty string when there is no prior history yet.
        """
        if not self.summary and not self.turns:
            return ""

        parts: list[str] = []

        if self.summary:
            parts.append(f"[Summary of earlier conversation]\n{self.summary}")

        for human, ai in self.turns:
            parts.append(f"Human: {human}\nAI: {ai}")

        return "\n\n".join(parts)

    def clear(self) -> None:
        """Reset to an empty state (equivalent to creating a new instance)."""
        self.summary = ""
        self.turns = []

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _verbatim_chars(self) -> int:
        return sum(len(h) + len(a) for h, a in self.turns)

    def _maybe_compress(self) -> None:
        """Compress oldest half of turns into the running summary if over budget."""
        if self._verbatim_chars() <= self.max_chars:
            return

        # Compress the oldest half; keep the newer half verbatim.
        split = max(1, len(self.turns) // 2)
        to_compress = self.turns[:split]
        self.turns = self.turns[split:]

        new_lines = "\n".join(
            f"Human: {h}\nAI: {a}" for h, a in to_compress
        )
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
            prompt = _SUMMARY_PROMPT.format(
                summary=self.summary, new_lines=new_lines
            )
            result = llm.invoke([HumanMessage(content=prompt)])
            self.summary = result.content.strip()
            logger.debug(
                "SessionMemory compressed %d turn(s) → summary length=%d",
                split, len(self.summary),
            )
        except Exception as exc:  # noqa: BLE001
            # Compression failed — just concatenate into summary as plain text
            # so we don't silently lose context.
            logger.warning("SessionMemory compression LLM call failed: %s", exc)
            if self.summary:
                self.summary = f"{self.summary}\n\n{new_lines}"
            else:
                self.summary = new_lines
