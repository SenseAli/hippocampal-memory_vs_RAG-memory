"""Consolidated memory store: LLM-generated generalized summaries.

One row per consolidation cluster. Each carries provenance
(`source_episode_ids`) and a `confidence` score from the LLM.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .. import config
from ._base import _decode_metadata, _safe_metadata, get_collection


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConsolidatedStore:
    def __init__(self) -> None:
        self._coll = get_collection(config.CONSOLIDATED_COLL)

    def add(
        self,
        session_id: str,
        text: str,
        embedding: list[float],
        *,
        source_episode_ids: list[str],
        confidence: float,
        support_count: int,
        preserved_facts: list[str] | None = None,
    ) -> str:
        cid = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "kind": "consolidated",
            "session_id": session_id,
            "created_at": _now_iso(),
            "source_episode_ids": list(source_episode_ids),
            "source_turns": [],
            "support_count": int(support_count),
            "confidence": float(confidence),
            "preserved_facts": list(preserved_facts or []),
            "superseded_by": "",
        }
        self._coll.add(
            ids=[cid],
            documents=[text],
            embeddings=[embedding],
            metadatas=[_safe_metadata(metadata)],
        )
        return cid

    def query(
        self, session_id: str, embedding: list[float], k: int = 5
    ) -> list[dict]:
        n_results = min(max(k * 3, k), 100)
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
            out.append(
                {
                    "id": cid,
                    "text": doc,
                    "metadata": md,
                    "score": 1.0 - float(dist) if dist is not None else 0.0,
                    "recency": 0.5,  # consolidated has no turn_index; constant moderate recency
                }
            )
        out.sort(key=lambda h: h["score"], reverse=True)
        return out[:k]

    def list_for_session(self, session_id: str) -> list[dict]:
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
        out.sort(key=lambda h: h["metadata"].get("created_at", ""))
        return out

    def find_near_duplicate(
        self, session_id: str, embedding: list[float], threshold: float
    ) -> dict | None:
        """Return the existing consolidated memory with cosine >= threshold,
        or None. Used by the consolidator to prevent bloat."""
        res = self._coll.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"session_id": session_id},
        )
        if not res or not res.get("ids") or not res["ids"][0]:
            return None
        cid = res["ids"][0][0]
        dist = res["distances"][0][0]
        score = 1.0 - float(dist)
        if score < threshold:
            return None
        return {
            "id": cid,
            "text": res["documents"][0][0],
            "metadata": _decode_metadata(res["metadatas"][0][0] or {}),
            "score": score,
        }

    def merge_into(
        self, target_id: str, source_ids: list[str], new_summary_text: str
    ) -> None:
        """Fold extra source_episode_ids into an existing consolidated memory
        and bump its support_count."""
        existing = self._coll.get(
            ids=[target_id], include=["metadatas", "documents"]
        )
        if not existing.get("ids"):
            return
        meta = _decode_metadata(existing["metadatas"][0] or {})
        prev_sources = list(meta.get("source_episode_ids", []) or [])
        merged = list(dict.fromkeys(prev_sources + source_ids))
        meta["source_episode_ids"] = merged
        meta["support_count"] = int(meta.get("support_count", 0)) + len(source_ids)
        # Keep the longer / newer text; caller decides.
        self._coll.update(
            ids=[target_id],
            documents=[new_summary_text],
            metadatas=[_safe_metadata(meta)],
        )

    def count(self, session_id: str | None = None) -> int:
        if session_id is None:
            return self._coll.count()
        return len(self.list_for_session(session_id))

    def get(self, memory_id: str) -> dict | None:
        res = self._coll.get(
            ids=[memory_id], include=["documents", "metadatas"]
        )
        if not res.get("ids"):
            return None
        return {
            "id": res["ids"][0],
            "text": res["documents"][0],
            "metadata": _decode_metadata(res["metadatas"][0] or {}),
        }
