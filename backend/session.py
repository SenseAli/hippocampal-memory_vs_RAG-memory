"""Session manager: in-memory dict + debounced JSON persistence.

Embeddings live in Chroma (always persisted). The Session JSON only carries
the lightweight chat history and bookkeeping — enough to hydrate the UI
after a reload, with the canonical memory in the vector store.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TurnRecord:
    turn_index: int
    user_text: str
    hippocampal_reply: str | None = None
    naive_reply: str | None = None
    hippocampal_retrieved: list[dict] = field(default_factory=list)
    naive_retrieved: list[dict] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    created_at: str
    turns: list[TurnRecord] = field(default_factory=list)
    last_consolidation: str | None = None
    last_turn_tokens_retrieved: int = 0
    _dirty: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "turns": [asdict(t) for t in self.turns],
            "last_consolidation": self.last_consolidation,
            "last_turn_tokens_retrieved": self.last_turn_tokens_retrieved,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        turns = [TurnRecord(**t) for t in d.get("turns", [])]
        s = cls(
            session_id=d["session_id"],
            created_at=d.get("created_at", _now_iso()),
            turns=turns,
            last_consolidation=d.get("last_consolidation"),
            last_turn_tokens_retrieved=int(d.get("last_turn_tokens_retrieved", 0)),
        )
        return s

    def mark_dirty(self) -> None:
        self._dirty = True


class SessionManager:
    """Thread-safe in-memory map of sessions with debounced disk writes."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._dir = persist_dir or config.SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._write_tasks: dict[str, asyncio.Task] = {}

    # --- lifecycle ---

    def create(self) -> Session:
        sid = str(uuid.uuid4())
        s = Session(session_id=sid, created_at=_now_iso())
        with self._lock:
            self._sessions[sid] = s
        self._persist(s)
        return s

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is not None:
                return s
        # Try disk.
        path = self._path_for(session_id)
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                s = Session.from_dict(d)
                with self._lock:
                    self._sessions[session_id] = s
                return s
            except Exception as e:  # pragma: no cover
                log.warning("Failed to load session %s: %s", session_id, e)
        return None

    def get_or_404(self, session_id: str) -> Session:
        s = self.get(session_id)
        if s is None:
            raise KeyError(session_id)
        return s

    def list_ids(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.json")]

    # --- mutation ---

    def add_turn(
        self,
        session_id: str,
        user_text: str,
        hippocampal_reply: str | None = None,
        naive_reply: str | None = None,
        hippocampal_retrieved: list[dict] | None = None,
        naive_retrieved: list[dict] | None = None,
    ) -> TurnRecord:
        s = self.get_or_404(session_id)
        with self._lock:
            idx = len(s.turns)
            tr = TurnRecord(
                turn_index=idx,
                user_text=user_text,
                hippocampal_reply=hippocampal_reply,
                naive_reply=naive_reply,
                hippocampal_retrieved=list(hippocampal_retrieved or []),
                naive_retrieved=list(naive_retrieved or []),
            )
            s.turns.append(tr)
            s.mark_dirty()
        self._persist(s)
        return tr

    def update_turn(
        self,
        session_id: str,
        turn_index: int,
        *,
        hippocampal_reply: str | None = None,
        naive_reply: str | None = None,
        hippocampal_retrieved: list[dict] | None = None,
        naive_retrieved: list[dict] | None = None,
        tokens_retrieved: int | None = None,
    ) -> None:
        s = self.get_or_404(session_id)
        with self._lock:
            if turn_index < 0 or turn_index >= len(s.turns):
                return
            tr = s.turns[turn_index]
            if hippocampal_reply is not None:
                tr.hippocampal_reply = hippocampal_reply
            if naive_reply is not None:
                tr.naive_reply = naive_reply
            if hippocampal_retrieved is not None:
                tr.hippocampal_retrieved = list(hippocampal_retrieved)
            if naive_retrieved is not None:
                tr.naive_retrieved = list(naive_retrieved)
            if tokens_retrieved is not None:
                s.last_turn_tokens_retrieved = int(tokens_retrieved)
            s.mark_dirty()
        self._persist(s)

    def set_last_consolidation(self, session_id: str, iso_ts: str) -> None:
        s = self.get_or_404(session_id)
        with self._lock:
            s.last_consolidation = iso_ts
            s.mark_dirty()
        self._persist(s)

    # --- persistence ---

    def _path_for(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def _persist(self, session: Session) -> None:
        # Synchronous write. Sessions are small; debouncing would be premature
        # optimisation at this scale.
        try:
            path = self._path_for(session.session_id)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception as e:  # pragma: no cover
            log.warning("Failed to persist session %s: %s", session.session_id, e)
