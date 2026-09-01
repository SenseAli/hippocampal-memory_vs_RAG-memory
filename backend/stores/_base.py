"""Shared ChromaDB PersistentClient + helpers used by every store."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from .. import config


_client = None


def get_chroma_client() -> chromadb.api.ClientAPI:
    """One persistent client per process. Chroma is happy with this."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def _safe_metadata(m: dict[str, Any] | None) -> dict[str, Any]:
    """Chroma requires metadata values to be str/int/float/bool. Cast lists
    to JSON strings; everything else pass through."""
    import json

    if not m:
        return {}
    out: dict[str, Any] = {}
    for k, v in m.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[k] = json.dumps(v)
        else:
            out[k] = str(v)
    return out


def _decode_metadata(m: dict[str, Any] | None) -> dict[str, Any]:
    """Reverse of _safe_metadata: JSON-decode list-typed fields."""
    import json

    if not m:
        return {}
    out: dict[str, Any] = dict(m)
    for k, v in list(out.items()):
        if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    return out


def get_collection(name: str) -> Collection:
    """Get-or-create a collection. Used at startup by each store."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # cosine distance, not L2
    )
