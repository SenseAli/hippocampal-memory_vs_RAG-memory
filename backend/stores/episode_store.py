"""Episode store: per-turn records for the hippocampal agent.

Two episodes per turn (user + assistant). Soft-delete via `superseded_by`
on consolidation. Raw episodes are never hard-deleted so the user can
always inspect them via the stats/debug surface.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .. import config
from ._base import _decode_metadata, _safe_metadata, get_collection


@dataclass
class Episode:
    id: str
    session_id: str
    turn_index: int
    role: str  # "user" | "assistant"
    text: str
    metadata: dict

    @property
    def embedding(self) -> list[float] | None:
        # Embeddings are stored in Chroma, not on the dataclass.
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeStore:
    def __init__(self) -> None:
        self._coll = get_collection(config.EPISODE_COLL)

    # --- writes ---

    def add(
        self,
        session_id: str,
        turn_index: int,
        role: str,
        text: str,
        embedding: list[float],
        *,
        user_query: str = "",
        source_chunk_ids: list[str] | None = None,
        consolidated_ids: list[str] | None = None,
        extra_metadata: dict | None = None,
        tokens: int = 0,
    ) -> str:
        eid = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "kind": "episode",
            "session_id": session_id,
            "turn_index": int(turn_index),
            "role": role,
            "created_at": _now_iso(),
            "tokens": int(tokens),
            "superseded_by": "",
            "user_query": user_query,
            "source_chunk_ids": list(source_chunk_ids or []),
            "consolidated_ids": list(consolidated_ids or []),
        }
        if extra_metadata:
            for k, v in extra_metadata.items():
                if k not in metadata:
                    metadata[k] = v
        self._coll.add(
            ids=[eid],
            documents=[text],
            embeddings=[embedding],
            metadatas=[_safe_metadata(metadata)],
        )
        return eid

    def mark_superseded(self, episode_ids: list[str], by_consolidated_id: str) -> None:
        """Soft-delete: set `superseded_by` on a batch of episodes."""
        if not episode_ids:
            return
        # Chroma requires a `where` filter or we get all rows. Get current meta first.
        existing = self._coll.get(ids=episode_ids, include=["metadatas"])
        new_metas = []
        for i, meta in enumerate(existing.get("metadatas", [])):
            m = _decode_metadata(meta or {})
            m["superseded_by"] = by_consolidated_id
            new_metas.append(_safe_metadata(m))
        if not new_metas:
            return
        # Chroma `update` wants parallel lists of ids + metadatas; order is preserved.
        self._coll.update(ids=list(existing["ids"]), metadatas=new_metas)

    def update_metadata(self, episode_id: str, patch: dict) -> None:
        existing = self._coll.get(ids=[episode_id], include=["metadatas"])
        if not existing.get("ids"):
            return
        meta = _decode_metadata(existing["metadatas"][0] or {})
        meta.update(patch)
        self._coll.update(
            ids=[episode_id], metadatas=[_safe_metadata(meta)]
        )

    # --- reads ---

    def query(
        self,
        session_id: str,
        embedding: list[float],
        k: int = 8,
        current_turn: int = 0,
    ) -> list[dict]:
        """Top-k episodes for a session, with active (non-superseded) filter
        applied at the Chroma layer where possible, then post-filtered for
        safety."""
        # We over-fetch to compensate for any superseded rows we filter out.
        n_results = min(max(k * 3, k), 200)
        res = self._coll.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where={"session_id": session_id},
        )
        out: list[dict] = []
        if not res or not res.get("ids"):
            return out
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            md = _decode_metadata(meta or {})
            if md.get("superseded_by"):
                continue
            if current_turn and md.get("turn_index") is not None:
                # Recency: more recent = higher. Distance from current turn.
                recency = 1.0 / (1.0 + max(0, current_turn - int(md["turn_index"])))
            else:
                recency = 0.0
            out.append(
                {
                    "id": cid,
                    "text": doc,
                    "metadata": md,
                    "score": 1.0 - float(dist) if dist is not None else 0.0,
                    "recency": recency,
                }
            )
        out.sort(key=lambda h: h["score"], reverse=True)
        return out[:k]

    def list_active(self, session_id: str) -> list[dict]:
        """All active (non-superseded) episodes for a session, oldest first."""
        res = self._coll.get(
            where={"session_id": session_id}, include=["documents", "metadatas"]
        )
        out: list[dict] = []
        for cid, doc, meta in zip(
            res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
        ):
            md = _decode_metadata(meta or {})
            if md.get("superseded_by"):
                continue
            out.append({"id": cid, "text": doc, "metadata": md})
        out.sort(key=lambda h: int(h["metadata"].get("turn_index", 0)))
        return out

    def get(self, episode_id: str) -> dict | None:
        res = self._coll.get(ids=[episode_id], include=["documents", "metadatas"])
        if not res.get("ids"):
            return None
        return {
            "id": res["ids"][0],
            "text": res["documents"][0],
            "metadata": _decode_metadata(res["metadatas"][0] or {}),
        }

    def count(self, session_id: str | None = None) -> int:
        if session_id is None:
            return self._coll.count()
        return len(self.list_active(session_id))

    def count_all(self, session_id: str | None = None) -> int:
        """Count every episode for a session, including superseded ones."""
        if session_id is None:
            return self._coll.count()
        res = self._coll.get(where={"session_id": session_id}, include=[])
        return len(res.get("ids", []))
