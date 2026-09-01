"""FastAPI app: HTTP + SSE endpoints, session lifecycle, both agents wired.

Endpoints
---------
GET    /                              -> serve frontend/index.html
GET    /static/{path:path}            -> serve frontend assets
POST   /api/session                   -> create session
GET    /api/session/{id}              -> full session state (for UI hydration)
POST   /api/upload                    -> multipart file upload
POST   /api/message/stream            -> SSE stream: agent reply + retrieved
POST   /api/consolidate               -> run sleep pass
GET    /api/stats                     -> memory stats for the UI
GET    /api/health                    -> liveness probe
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from . import config
from .agents.hippocampal import HippocampalAgent
from .agents.naive_rag import NaiveRAGAgent
from .embeddings import Embedder
from .llm import client as ollama_client
from .models import (
    ConsolidationReport,
    SessionOut,
    SessionState,
    StatsOut,
    Turn,
    UploadOut,
)
from .pipeline.consolidate import consolidate
from .pipeline.ingest import ingest_file
from .session import SessionManager
from .stores.chunk_store import ChunkStore
from .stores.consolidated_store import ConsolidatedStore
from .stores.episode_store import EpisodeStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("backend.main")


# --- App + singletons ---


def create_app() -> FastAPI:
    app = FastAPI(title="Hippocampal Agent Demo", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    embedder = Embedder.instance()
    chunk_store = ChunkStore()
    episode_store = EpisodeStore()
    consolidated_store = ConsolidatedStore()
    sessions = SessionManager()

    hippo_agent = HippocampalAgent(
        episode_store=episode_store,
        consolidated_store=consolidated_store,
        embedder=embedder,
        llm=ollama_client,
        session_manager=sessions,
    )
    naive_agent = NaiveRAGAgent(
        chunk_store=chunk_store,
        embedder=embedder,
        llm=ollama_client,
        session_manager=sessions,
    )

    # Stash on app.state for testability.
    app.state.embedder = embedder
    app.state.chunk_store = chunk_store
    app.state.episode_store = episode_store
    app.state.consolidated_store = consolidated_store
    app.state.sessions = sessions
    app.state.hippo_agent = hippo_agent
    app.state.naive_agent = naive_agent

    # --- Static / frontend ---
    frontend_dir = config.FRONTEND_DIR
    if (frontend_dir / "index.html").exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(frontend_dir)),
            name="static",
        )

        @app.get("/")
        def root() -> FileResponse:
            return FileResponse(frontend_dir / "index.html")

    # --- Health ---

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "ollama_host": config.OLLAMA_HOST,
            "chat_model": config.CHAT_MODEL,
            "embed_model": config.EMBED_MODEL,
        }

    # --- Sessions ---

    @app.post("/api/session", response_model=SessionOut)
    def post_session() -> SessionOut:
        s = sessions.create()
        return SessionOut(session_id=s.session_id, created_at=s.created_at)

    @app.get("/api/session/{session_id}", response_model=SessionState)
    def get_session(session_id: str) -> SessionState:
        s = sessions.get_or_404(session_id)
        turns = [
            Turn(
                turn_index=t.turn_index,
                user_text=t.user_text,
                hippocampal_reply=t.hippocampal_reply,
                naive_reply=t.naive_reply,
            )
            for t in s.turns
        ]
        return SessionState(
            session_id=s.session_id,
            created_at=s.created_at,
            turns=turns,
            episodic_count=episode_store.count(s.session_id),
            consolidated_count=consolidated_store.count(s.session_id),
            last_consolidation=s.last_consolidation,
        )

    # --- Upload ---

    @app.post("/api/upload", response_model=UploadOut)
    async def post_upload(
        session_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> UploadOut:
        # Make sure the session exists (so we get clean 404s).
        sessions.get_or_404(session_id)

        # Persist the upload to a temp file so ingest_file can path-open it.
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (".txt", ".md", ".markdown", ".pdf"):
            raise HTTPException(400, f"Unsupported file type: {suffix!r}")

        tmp_path = Path(config.DATA_DIR) / f"upload_{int(time.time())}{suffix}"
        try:
            data = await file.read()
            tmp_path.write_bytes(data)
            result = ingest_file(tmp_path, chunk_store, embedder)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        return UploadOut(
            doc_id=result["doc_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            total_tokens=result["total_tokens"],
        )

    # --- Message stream (SSE) ---

    @app.post("/api/message/stream")
    async def post_message_stream(
        request: Request,
    ) -> StreamingResponse:
        """Stream a reply for one agent. Body: {session_id, text, agent}."""
        body = await request.json()
        session_id = body.get("session_id")
        text = body.get("text") or ""
        agent = body.get("agent") or "hippocampal"
        if not session_id or not text:
            raise HTTPException(400, "session_id and text required")
        if agent not in ("hippocampal", "naive"):
            raise HTTPException(400, f"unknown agent: {agent!r}")

        # We start a turn record up front so the UI sees a turn in the history
        # while the assistant is streaming. The reply field is filled in as
        # the final SSE event arrives.
        turn = sessions.add_turn(
            session_id=session_id,
            user_text=text,
            hippocampal_reply=None if agent == "hippocampal" else "",
            naive_reply=None if agent == "naive" else "",
        )

        async def event_source() -> AsyncIterator[bytes]:
            queue: asyncio.Queue[dict | None] = asyncio.Queue()

            def on_token(tok: str) -> None:
                # Schedule the put on the running event loop.
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon_threadsafe(queue.put_nowait, {"t": tok})
                except RuntimeError:
                    pass

            def run_agent() -> None:
                try:
                    if agent == "hippocampal":
                        result = hippo_agent.turn(
                            session_id, text, on_token=on_token
                        )
                        sessions.update_turn(
                            session_id,
                            turn.turn_index,
                            hippocampal_reply=result.reply,
                            hippocampal_retrieved=result.retrieved,
                            tokens_retrieved=result.tokens_retrieved,
                        )
                    else:
                        result = naive_agent.turn(
                            session_id, text, on_token=on_token
                        )
                        sessions.update_turn(
                            session_id,
                            turn.turn_index,
                            naive_reply=result.reply,
                            naive_retrieved=result.retrieved,
                            tokens_retrieved=result.tokens_retrieved,
                        )
                    # Send the final "done" payload.
                    asyncio.run_coroutine_threadsafe(
                        queue.put(
                            {
                                "done": True,
                                "retrieved": result.retrieved,
                                "tokens_retrieved": result.tokens_retrieved,
                                "agent": agent,
                                "turn_index": turn.turn_index,
                            }
                        ),
                        loop,
                    )
                except Exception as e:  # pragma: no cover
                    log.exception("Agent %s failed: %s", agent, e)
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"error": str(e)}),
                        loop,
                    )
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            loop = asyncio.get_running_loop()
            # Emit a "starting" event with the turn index so the UI can
            # render the user bubble immediately.
            yield _sse(
                {
                    "type": "start",
                    "turn_index": turn.turn_index,
                    "agent": agent,
                }
            )
            # Run the agent in a background thread; stream tokens back.
            fut = loop.run_in_executor(None, run_agent)
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if "t" in item:
                        yield _sse({"type": "token", "data": item["t"]})
                    elif item.get("done"):
                        yield _sse(
                            {
                                "type": "done",
                                "retrieved": item["retrieved"],
                                "tokens_retrieved": item["tokens_retrieved"],
                                "agent": item["agent"],
                                "turn_index": item["turn_index"],
                            }
                        )
                    elif "error" in item:
                        yield _sse({"type": "error", "data": item["error"]})
            finally:
                # Make sure the executor thread doesn't dangle.
                try:
                    await fut
                except Exception:
                    pass

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # --- Consolidate ---

    @app.post("/api/consolidate", response_model=ConsolidationReport)
    def post_consolidate(body: dict) -> ConsolidationReport:
        session_id = body.get("session_id")
        if not session_id:
            raise HTTPException(400, "session_id required")
        sessions.get_or_404(session_id)
        report = consolidate(
            session_id,
            episode_store,
            consolidated_store,
            embedder,
            ollama_client,
        )
        if report.get("memories_created", 0) > 0:
            sessions.set_last_consolidation(
                session_id, datetime.now(timezone.utc).isoformat()
            )
        return ConsolidationReport(**report)

    # --- Stats ---

    @app.get("/api/stats", response_model=StatsOut)
    def get_stats(session_id: str) -> StatsOut:
        s = sessions.get_or_404(session_id)
        epi_active = episode_store.count(s.session_id)
        epi_total = episode_store.count_all(s.session_id)
        epi_superseded = max(0, epi_total - epi_active)
        return StatsOut(
            session_id=s.session_id,
            episodic_count=epi_active,
            episodic_superseded=epi_superseded,
            consolidated_count=consolidated_store.count(s.session_id),
            last_consolidation=s.last_consolidation,
            last_turn_tokens_retrieved=s.last_turn_tokens_retrieved,
            chunk_count=chunk_store.count(),
        )

    return app


def _sse(payload: dict) -> bytes:
    """Encode a dict as one SSE event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


app = create_app()
