"""Centralised configuration. Override any of these via env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    # backend/config.py -> backend/ -> project root
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _project_root()
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data")))
CHROMA_DIR: Path = DATA_DIR / "chroma"
SESSIONS_DIR: Path = DATA_DIR / "sessions"
FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"

# Ensure runtime directories exist.
for _d in (DATA_DIR, CHROMA_DIR, SESSIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Models ---
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL: str = os.environ.get("CHAT_MODEL", "llama3")
EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

# --- Chroma collection names ---
CHUNK_COLL: str = "chunks"
EPISODE_COLL: str = "episodes"
CONSOLIDATED_COLL: str = "consolidated_memories"

# --- Retrieval / consolidation thresholds ---
SIM_THRESHOLD: float = float(os.environ.get("SIM_THRESHOLD", "0.78"))
CONSOLIDATED_DEDUP_THRESHOLD: float = float(
    os.environ.get("CONSOLIDATED_DEDUP_THRESHOLD", "0.9")
)
MIN_CLUSTER_SIZE: int = 2
MAX_CLUSTERS_PER_PASS: int = 6
MAX_EPISODES_PER_PASS: int = 40
FORCE_SUPERSEDE_MIN_SUPPORT: int = 3
FORCE_SUPERSEDE_MIN_CONFIDENCE: float = 0.7

# --- Per-turn retrieval ---
K_CONSOLIDATED: int = 5
K_EPISODES: int = 8
FINAL_TOP_N: int = 8
CHUNK_TOP_K: int = 8

# --- Scoring weights (must sum to 1.0) ---
CONSOLIDATED_COSINE_W: float = 0.7
CONSOLIDATED_CONFIDENCE_W: float = 0.2
CONSOLIDATED_RECENCY_W: float = 0.1
EPISODE_COSINE_W: float = 0.6
EPISODE_RECENCY_W: float = 0.4

# --- Token budget (rough; tiktoken is approximate) ---
BUDGET_SYSTEM: int = 350
BUDGET_CHAT_HISTORY: int = 1500
BUDGET_MEMORIES: int = 1800
BUDGET_USER_MSG: int = 600
PER_MEMORY_CAP_TOKENS: int = 400


# --- Ingest chunking ---
CHUNK_SIZE_TOKENS: int = 500
CHUNK_OVERLAP_TOKENS: int = 50


@dataclass(frozen=True)
class RetrievalTuning:
    """Snapshot of the knobs that affect retrieval behaviour."""

    sim_threshold: float = SIM_THRESHOLD
    k_consolidated: int = K_CONSOLIDATED
    k_episodes: int = K_EPISODES
    final_top_n: int = FINAL_TOP_N


def frontend_index_path() -> Path:
    return FRONTEND_DIR / "index.html"
