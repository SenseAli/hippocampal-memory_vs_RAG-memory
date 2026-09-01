"""Naive RAG agent. Flat top-k chunk retrieval + sliding chat window.

This agent is the comparison baseline. It deliberately:
  - does NOT maintain any memory of past turns beyond the recent chat window
  - does NOT do consolidation
  - retrieves only document chunks, not anything that was said in earlier turns

This is what makes the side-by-side comparison meaningful.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .. import config
from ..embeddings import Embedder
from ..llm import OllamaClient
from ..pipeline.context import (
    RankedHit,
    build_naive_rag_messages,
)
from ..session import Session, SessionManager
from ..stores.chunk_store import ChunkStore
from .prompts import NAIVE_RAG_SYSTEM_PROMPT


log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    reply: str
    retrieved: list[dict]  # serialized hits for the UI
    tokens_retrieved: int
    messages: list[dict]  # the LLM call payload (for debug)


def _session_recent_turns(session: Session, max_turns: int = 6) -> list[dict]:
    """Return last N (user, assistant) pairs as OpenAI-style messages, but
    we just want role+text dicts."""
    out: list[dict] = []
    for t in session.turns[-max_turns:]:
        out.append({"role": "user", "text": t.user_text})
        if t.hippocampal_reply is not None or t.naive_reply is not None:
            out.append(
                {
                    "role": "assistant",
                    "text": t.naive_reply or t.hippocampal_reply or "",
                }
            )
    return out


class NaiveRAGAgent:
    def __init__(
        self,
        chunk_store: ChunkStore,
        embedder: Embedder,
        llm: OllamaClient,
        session_manager: SessionManager,
    ) -> None:
        self._chunks = chunk_store
        self._embedder = embedder
        self._llm = llm
        self._sessions = session_manager

    def turn(
        self,
        session_id: str,
        user_message: str,
        *,
        on_token=None,
    ) -> TurnResult:
        session = self._sessions.get_or_404(session_id)
        q_vec = self._embedder.encode_one(user_message)

        # Retrieve top-k chunks.
        chunk_hits = self._chunks.query(q_vec, k=config.CHUNK_TOP_K)
        ranked = [
            RankedHit(
                id=h["id"],
                kind="chunk",
                text=h["text"],
                score=h["score"],
                recency=0.0,
                final_score=h["score"],
                extra={"metadata": h.get("metadata", {})},
            )
            for h in chunk_hits
        ]

        recent = _session_recent_turns(session)
        messages, used_tokens = build_naive_rag_messages(
            system_prompt=NAIVE_RAG_SYSTEM_PROMPT,
            recent_turns=recent,
            chunks=ranked,
            user_message=user_message,
        )

        if on_token is not None:
            reply = self._llm.chat(messages, stream=True, on_token=on_token)
        else:
            reply = self._llm.chat(messages, stream=False)

        # Serialize retrieved for UI.
        retrieved_serialized = [
            {
                "id": h.id,
                "kind": h.kind,
                "text": h.text,
                "score": h.score,
                "final_score": h.final_score,
                "recency": h.recency,
                "turn_index": h.turn_index,
                "role": h.role,
                "snippet": h.text[:240],
            }
            for h in ranked
        ]

        return TurnResult(
            reply=reply,
            retrieved=retrieved_serialized,
            tokens_retrieved=used_tokens,
            messages=messages,
        )
