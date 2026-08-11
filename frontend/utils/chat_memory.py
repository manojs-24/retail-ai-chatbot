from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


_DEFAULT_MAX_CHARS = 3_000


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

    # Public API

    def save_context(self, human: str, ai: str) -> None:
        self.turns.append((human, ai))
        self._maybe_compress()

    def load_context(self) -> str:
        if not self.summary and not self.turns:
            return ""

        parts: list[str] = []

        if self.summary:
            parts.append(f"[Summary of earlier conversation]\n{self.summary}")

        for human, ai in self.turns:
            parts.append(f"Human: {human}\nAI: {ai}")

        return "\n\n".join(parts)

    def clear(self) -> None:
        self.summary = ""
        self.turns = []

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def _verbatim_chars(self) -> int:
        return sum(len(h) + len(a) for h, a in self.turns)

    def _maybe_compress(self) -> None:
        if self._verbatim_chars() <= self.max_chars:
            return

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
        except Exception as exc:
            logger.warning("SessionMemory compression LLM call failed: %s", exc)
            if self.summary:
                self.summary = f"{self.summary}\n\n{new_lines}"
            else:
                self.summary = new_lines
