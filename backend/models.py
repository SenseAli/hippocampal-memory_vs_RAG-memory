"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --- Sessions ---


class SessionOut(BaseModel):
    session_id: str
    created_at: str


# --- Upload ---


class UploadOut(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    total_tokens: int


# --- Messages ---


class MemoryHit(BaseModel):
    id: str
    kind: Literal["episode", "consolidated", "chunk"]
    text: str
    score: float
    recency: float = 0.0
    final_score: float = 0.0
    turn_index: int | None = None
    role: str | None = None
    session_id: str | None = None
    snippet: str = ""


class MessageIn(BaseModel):
    session_id: str
    text: str
    agent: Literal["hippocampal", "naive"] = "hippococampal"


# --- Stats ---


class StatsOut(BaseModel):
    session_id: str
    episodic_count: int
    episodic_superseded: int
    consolidated_count: int
    last_consolidation: str | None
    last_turn_tokens_retrieved: int
    chunk_count: int


# --- Consolidation report ---


class ConsolidationReport(BaseModel):
    session_id: str
    clusters_processed: int
    memories_created: int
    episodes_marked_superseded: int
    skipped: str | None = None
    duration_ms: int
    details: list[dict] = Field(default_factory=list)


# --- Session hydration (for page reload) ---


class Turn(BaseModel):
    turn_index: int
    user_text: str
    hippocampal_reply: str | None = None
    naive_reply: str | None = None


class SessionState(BaseModel):
    session_id: str
    created_at: str
    turns: list[Turn]
    episodic_count: int
    consolidated_count: int
    last_consolidation: str | None
