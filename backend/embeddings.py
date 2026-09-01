"""Thin wrapper around sentence-transformers with lazy model loading."""

from __future__ import annotations

import threading
from typing import Iterable

import numpy as np

from . import config


class Embedder:
    """Lazily-loaded sentence-transformers model. Thread-safe single instance."""

    _instance: "Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or config.EMBED_MODEL
        self._model = None  # loaded on first use

    @classmethod
    def instance(cls) -> "Embedder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_loaded(self) -> None:
        if self._model is None:
            # Import inside the function so module import stays cheap.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            # Pin embedding dimension for downstream code.
            dim = self._model.get_embedding_dimension()
            self._dim = int(dim) if dim is not None else 0

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return self._dim

    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        self._ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        # convert_to_numpy=True is the default but be explicit.
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine sim becomes dot product
        )
        return vectors.astype(np.float32)

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()
