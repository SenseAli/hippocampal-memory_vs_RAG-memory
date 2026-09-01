"""Hippocampal agent: per-turn pipeline that retrieves from both memory tiers.

The per-turn flow is:
  1. embed user query
  2. retrieve from consolidated_memories (K_c)
  3. retrieve from episodes (K_e)
  4. score each hit (cosine + recency, weighted by kind)
  5. merge, take top N
  6. build token-budgeted context
  7. call LLM
  8. write two new episodes: user message + assistant reply

This is the agent that "remembers" prior turns and consolidates them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .. import config
from ..embeddings import Embedder
from ..llm import OllamaClient
from ..pipeline.context import (
    RankedHit,
    build_hippocampal_messages,
    count_tokens,
)
from ..pipeline.ingest import _encoder  # reused for token counts
from ..session import SessionManager
from ..stores.consolidated_store import ConsolidatedStore
from ..stores.episode_store import EpisodeStore
from .prompts import HIPPOCAMPAL_SYSTEM_PROMPT


log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    reply: str
    retrieved: list[dict]
    tokens_retrieved: int
    messages: list[dict]
    user_episode_id: str
    assistant_episode_id: str


def _session_recent_turns(session, max_pairs: int = 6) -> list[dict]:
    out: list[dict] = []
    for t in session.turns[-max_pairs:]:
        out.append({"role": "user", "text": t.user_text})
        if t.hippocampal_reply is not None or t.naive_reply is not None:
            out.append(
                {
                    "role": "assistant",
                    "text": t.hippocampal_reply or t.naive_reply or "",
                }
            )
    return out


def _score_consolidated(cosine: float, confidence: float, recency: float) -> float:
    return (
        config.CONSOLIDATED_COSINE_W * cosine
        + config.CONSOLIDATED_CONFIDENCE_W * confidence
        + config.CONSOLIDATED_RECENCY_W * recency
    )


def _score_episode(cosine: float, recency: float) -> float:
    return config.EPISODE_COSINE_W * cosine + config.EPISODE_RECENCY_W * recency


class HippocampalAgent:
    def __init__(
        self,
        episode_store: EpisodeStore,
        consolidated_store: ConsolidatedStore,
        embedder: Embedder,
        llm: OllamaClient,
        session_manager: SessionManager,
    ) -> None:
        self._episodes = episode_store
        self._consolidated = consolidated_store
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
        current_turn = len(session.turns)

        # 1. Embed query.
        q_vec = self._embedder.encode_one(user_message)

        # 2. Retrieve from consolidated store.
        cons_hits_raw = self._consolidated.query(
            session_id, q_vec, k=config.K_CONSOLIDATED
        )
        cons_ranked: list[RankedHit] = []
        for h in cons_hits_raw:
            confidence = float(h["metadata"].get("confidence", 1.0))
            final = _score_consolidated(h["score"], confidence, h["recency"])
            cons_ranked.append(
                RankedHit(
                    id=h["id"],
                    kind="consolidated",
                    text=h["text"],
                    score=h["score"],
                    recency=h["recency"],
                    final_score=final,
                    extra={
                        "metadata": h["metadata"],
                        "support_count": h["metadata"].get("support_count", 0),
                    },
                )
            )

        # 3. Retrieve from episodes.
        epi_hits_raw = self._episodes.query(
            session_id,
            q_vec,
            k=config.K_EPISODES,
            current_turn=current_turn,
        )
        epi_ranked: list[RankedHit] = []
        for h in epi_hits_raw:
            final = _score_episode(h["score"], h["recency"])
            md = h["metadata"]
            epi_ranked.append(
                RankedHit(
                    id=h["id"],
                    kind="episode",
                    text=h["text"],
                    score=h["score"],
                    recency=h["recency"],
                    final_score=final,
                    turn_index=int(md.get("turn_index", 0)),
                    role=md.get("role", "?"),
                    extra={"metadata": md},
                )
            )

        # 4. Merge and take top N.
        combined = cons_ranked + epi_ranked
        combined.sort(key=lambda h: h.final_score, reverse=True)
        top = combined[: config.FINAL_TOP_N]

        # 5. Build context.
        recent = _session_recent_turns(session)
        messages, used_tokens = build_hippocampal_messages(
            system_prompt=HIPPOCAMPAL_SYSTEM_PROMPT,
            recent_turns=recent,
            retrieved=top,
            user_message=user_message,
        )

        # 6. Call LLM.
        if on_token is not None:
            reply = self._llm.chat(messages, stream=True, on_token=on_token)
        else:
            reply = self._llm.chat(messages, stream=False)

        # 7. Write episodes.
        # The user episode is written with the *current* turn_index; the
        # assistant episode records which consolidated memories it cited.
        consolidated_ids_used = [
            h.id for h in top if h.kind == "consolidated"
        ]
        source_chunk_ids: list[str] = []  # not currently used; future-proof
        enc = _encoder()
        user_tokens = len(enc.encode(user_message))
        reply_tokens = len(enc.encode(reply))

        user_eid = self._episodes.add(
            session_id=session_id,
            turn_index=current_turn,
            role="user",
            text=user_message,
            embedding=q_vec,
            tokens=user_tokens,
        )
        assistant_eid = self._episodes.add(
            session_id=session_id,
            turn_index=current_turn,
            role="assistant",
            text=reply,
            embedding=self._embedder.encode_one(reply),
            user_query=user_message,
            consolidated_ids=consolidated_ids_used,
            source_chunk_ids=source_chunk_ids,
            tokens=reply_tokens,
        )

        # 8. Serialize retrieved for the UI.
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
                "metadata": h.extra.get("metadata", {}) if h.extra else {},
            }
            for h in top
        ]

        return TurnResult(
            reply=reply,
            retrieved=retrieved_serialized,
            tokens_retrieved=used_tokens,
            messages=messages,
            user_episode_id=user_eid,
            assistant_episode_id=assistant_eid,
        )
