# Hippocampus-Inspired Memory System for LLM Agents

A side-by-side demo of a hippocampus-style memory agent vs naive RAG, on a research / document-grounded chat. The hippocampal agent has two memory tiers (episodic + consolidated) and a manual "sleep" button that runs replay/consolidation. Naive RAG does flat top-k chunk retrieval with no memory of past turns.

The comparison is that the UI shows exactly what each agent retrieved per turn.

## Architecture Used

Local-first Python. sentence-transformers for embeddings (`all-MiniLM-L6-v2`). Ollama for chat (`llama3`, configurable). ChromaDB for vector storage (persistent on disk). FastAPI backend, single-file HTML/JS frontend (no build step). One `uvicorn` command to run.

Two Chroma collections back the hippocampal agent — `episodes` (per-turn detail) and `consolidated_memories` (generalized long-term, produced by the "sleep" pass). One shared collection `chunks` backs naive RAG. Every turn, the hippocampal agent retrieves from both memory tiers, scores hits with cosine + recency, assembles a budgeted context, calls the LLM, and writes two new episodes (user message + assistant reply). Consolidation runs on demand: single-link clusters similar episodes, calls the LLM to produce a generalized summary, and either supersedes the source episodes (soft-delete via `superseded_by`) or coexists with them.

## To Run

Prereqs:
- Python 3.10+
- [Ollama](https://ollama.com/) installed
- `ollama pull llama3` (or change `CHAT_MODEL` in `backend/config.py`)

**Note on Windows**: the Ollama desktop app auto-starts a background `ollama serve` on port 11434. If you run `ollama serve` manually you'll get `bind: Only one usage of each socket address...` don't worry about that it just means the service is already up. Skip the manual command.

```bash
cd hippocampal-agent
pip install -e .
./run.sh
```

Open `http://localhost:8000/`.

## Using the demo

1. Click **New session**.
2. Upload 2-3 small `.txt` / `.md` / `.pdf` files in the top bar.
3. Type a message in either column — both agents reply with the same input, side-by-side.
4. Click **show retrieved memories** under any assistant message to see exactly what each agent pulled in.
5. After 5+ turns, click **Consolidate (sleep)** in the hippocampal column. Watch the stats panel: `Episodes`, `Consolidated`, `Last sleep`, `Tokens retrieved (last turn)`.
6. Ask synthesis or "as we discussed" questions to see the hippocampal agent cite `[CONSOLIDATED]` or `[EPISODE]` memories that naive RAG cannot.

## Where hippocampal wins, where naive RAG ties

| Naive RAG ties | Hippocampal wins |
|---|---|
| Recent factual Qs answerable from corpus | Callbacks to turn > 6 ago |
| Single-doc lookup | Synthesis across turns/docs |
| Short sessions (<10 turns, no consolidation yet) | "As we discussed" questions, evolved thesis |
| Documents small enough to fit in context | Contradiction detection across turns/docs |

## Project layout

```
backend/
  main.py               # FastAPI app + routes
  config.py             # paths, model names, thresholds
  embeddings.py         # sentence-transformers wrapper
  llm.py                # Ollama HTTP client
  session.py            # Session + SessionManager (in-mem + on-disk)
  models.py             # Pydantic schemas
  stores/               # Chroma collection wrappers
  agents/               # hippocampal, naive_rag, prompts
  pipeline/             # ingest, consolidate, context
frontend/
  index.html            # single-file SPA
data/
  chroma/               # persistent ChromaDB
  sessions/             # per-session JSON snapshots
```

## Configuration

Override defaults via env vars (all read in `backend/config.py`):

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `CHAT_MODEL` | `llama3` | Ollama chat model |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `SIM_THRESHOLD` | `0.78` | cosine cutoff for episode clustering |
| `DATA_DIR` | `./data` | where chroma and sessions live |
