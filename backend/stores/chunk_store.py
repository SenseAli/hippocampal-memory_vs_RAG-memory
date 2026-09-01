"""Chunk store: shared naive-RAG document chunks. Not session-scoped."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

from .. import config
from ._base import _decode_metadata, _safe_metadata, get_collection


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


class ChunkStore:
    """One row per document chunk (~500 tokens). Shared across sessions."""

    def __init__(self) -> None:
        self._coll = get_collection(config.CHUNK_COLL)

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if not texts:
            return []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]
        # Chroma wants aligned parallel lists; if any metadata is missing,
        # fill with empty dicts.
        if len(metadatas) < len(texts):
            metadatas = metadatas + [{} for _ in range(len(texts) - len(metadatas))]
        safe_meta = [_safe_metadata(m) for m in metadatas]
        self._coll.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=safe_meta,
        )
        return ids

    def query(
        self, embedding: list[float], k: int = 8, where: dict | None = None
    ) -> list[dict]:
        """Return top-k chunks as dicts with id, text, metadata, score."""
        res = self._coll.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
        )
        out: list[dict] = []
        if not res or not res.get("ids"):
            return out
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(
                {
                    "id": cid,
                    "text": doc,
                    "metadata": _decode_metadata(meta),
                    # Chroma cosine distance = 1 - cosine similarity for unit vectors.
                    "score": 1.0 - float(dist) if dist is not None else 0.0,
                }
            )
        return out

    def count(self) -> int:
        return self._coll.count()
