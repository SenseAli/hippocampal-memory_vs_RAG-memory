"""Document ingest: load .txt/.md/.pdf, chunk to ~500 tokens, write to chunk_store."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Callable

import tiktoken

from .. import config
from ..embeddings import Embedder
from ..stores.chunk_store import ChunkStore


log = logging.getLogger(__name__)

_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        # cl100k_base is a safe default (used by GPT-4 / many open models);
        # we only need rough token counts for chunking, not exact.
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


# --- Loaders ---


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_markdown(path: Path) -> str:
    # Naive: treat as text. Good enough for chunking; markdown structure
    # (headers, lists) survives the chunker fine.
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception as e:  # pragma: no cover
            log.warning("PDF page %d extract failed: %s", i, e)
            txt = ""
        if txt.strip():
            parts.append(txt)
    return "\n\n".join(parts)


LOADERS: dict[str, Callable[[Path], str]] = {
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
}


# --- Chunking ---


def chunk_text(
    text: str,
    size_tokens: int = config.CHUNK_SIZE_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Sliding-window chunking in token space.

    Splits on sentence boundaries where possible; falls back to token windows.
    """
    enc = _encoder()
    tokens = enc.encode(text)
    if not tokens:
        return []
    if len(tokens) <= size_tokens:
        return [text.strip()]

    chunks: list[str] = []
    step = max(1, size_tokens - overlap_tokens)
    for start in range(0, len(tokens), step):
        end = min(start + size_tokens, len(tokens))
        window = tokens[start:end]
        chunk = enc.decode(window).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
    return chunks


# --- Ingest entry point ---


def ingest_file(
    path: Path,
    chunk_store: ChunkStore,
    embedder: Embedder,
) -> dict:
    """Load, chunk, embed, and persist a single file. Returns a summary dict."""
    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported file type: {suffix!r}. Supported: {sorted(LOADERS.keys())}"
        )

    text = loader(path)
    if not text.strip():
        return {
            "doc_id": "",
            "filename": path.name,
            "chunk_count": 0,
            "total_tokens": 0,
        }

    chunks = chunk_text(text)
    if not chunks:
        return {
            "doc_id": "",
            "filename": path.name,
            "chunk_count": 0,
            "total_tokens": 0,
        }

    embeddings = embedder.encode(chunks)
    doc_id = str(uuid.uuid4())
    ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": path.name,
            "chunk_index": i,
            "source_path": str(path),
        }
        for i in range(len(chunks))
    ]
    chunk_store.add(
        texts=chunks,
        embeddings=[v.tolist() for v in embeddings],
        metadatas=metadatas,
        ids=ids,
    )
    total_tokens = sum(_count_tokens(c) for c in chunks)
    return {
        "doc_id": doc_id,
        "filename": path.name,
        "chunk_count": len(chunks),
        "total_tokens": total_tokens,
    }
