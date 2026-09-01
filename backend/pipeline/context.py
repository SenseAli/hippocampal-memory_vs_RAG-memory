"""Token-budgeted context assembly.

Given a system prompt, recent chat history, a ranked list of retrieved
memories/chunks, and a user message, build a list of OpenAI-style messages
that fits within the configured token budget.

Truncation policy: drop lowest-scored memories first, then within a single
memory truncate to PER_MEMORY_CAP_TOKENS. Never drop the user message
(truncate with a marker instead if it overflows).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import tiktoken

from .. import config


_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoder().encode(text))


@dataclass
class RankedHit:
    """One retrieved memory or chunk, ranked."""

    id: str
    kind: str  # "episode" | "consolidated" | "chunk"
    text: str
    score: float
    recency: float = 0.0
    final_score: float = 0.0
    turn_index: int | None = None
    role: str | None = None
    extra: dict | None = None


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    enc = _encoder()
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens]) + " ..."


def _format_memories(hits: list[RankedHit]) -> str:
    """Render the numbered-memory block that goes into the LLM prompt."""
    if not hits:
        return "(none retrieved)"
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        if h.kind == "episode":
            turn = h.turn_index if h.turn_index is not None else "?"
            role = h.role or "?"
            label = f"[EPISODE #{i} | turn {turn} | {role}]"
        elif h.kind == "consolidated":
            label = f"[CONSOLIDATED #{i}]"
        else:
            label = f"[CHUNK #{i}]"
        lines.append(f"{label} (score={h.final_score:.2f})")
        lines.append(_truncate_to_tokens(h.text, config.PER_MEMORY_CAP_TOKENS))
        lines.append("")
    return "\n".join(lines).strip()


def _format_chunks(hits: list[RankedHit]) -> str:
    """Render the naive-RAG chunks block."""
    if not hits:
        return "(no relevant document excerpts found)"
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        lines.append(f"[chunk #{i}] (score={h.score:.2f})")
        lines.append(_truncate_to_tokens(h.text, config.PER_MEMORY_CAP_TOKENS))
        lines.append("")
    return "\n".join(lines).strip()


def build_hippocampal_messages(
    *,
    system_prompt: str,
    recent_turns: list[dict],  # [{"role": "user"|"assistant", "text": "..."}]
    retrieved: list[RankedHit],
    user_message: str,
) -> tuple[list[dict], int]:
    """Build the messages list for the hippocampal agent.

    Returns (messages, total_tokens_used). Messages have role + content
    in OpenAI chat format.
    """
    used = 0
    sys_tokens = count_tokens(system_prompt)
    # If the system prompt alone is over budget, truncate it.
    if sys_tokens > config.BUDGET_SYSTEM:
        system_prompt = _truncate_to_tokens(system_prompt, config.BUDGET_SYSTEM)
        sys_tokens = config.BUDGET_SYSTEM
    used += sys_tokens

    memories_text = _format_memories(retrieved)
    # Truncate the memories block to budget, lowest-scored first.
    ranked = sorted(retrieved, key=lambda h: h.final_score, reverse=True)
    kept: list[RankedHit] = []
    mem_tokens = 0
    for h in ranked:
        candidate_text = _format_memories(kept + [h])
        t = count_tokens(candidate_text)
        if mem_tokens + t <= config.BUDGET_MEMORIES:
            kept.append(h)
            mem_tokens = t
        else:
            break
    if len(kept) < len(ranked):
        # Append a note that some memories were dropped.
        kept.append(
            RankedHit(
                id="__truncation__",
                kind="chunk",
                text=f"({len(ranked) - len(kept)} additional memories omitted due to token budget)",
                score=0.0,
            )
        )
    memories_text = _format_memories(kept)
    used += count_tokens(memories_text)

    # Recent chat history: take as many recent turns as fit.
    chat_tokens = 0
    chat_messages: list[dict] = []
    for turn in reversed(recent_turns):
        block = {"role": turn["role"], "content": turn["text"]}
        t = count_tokens(block["content"]) + 4  # +4 for role + formatting
        if chat_tokens + t > config.BUDGET_CHAT_HISTORY:
            break
        chat_messages.append(block)
        chat_tokens += t
    chat_messages.reverse()
    used += chat_tokens

    # User message: hard truncate if absurd.
    user_text = _truncate_to_tokens(user_message, config.BUDGET_USER_MSG)
    used += count_tokens(user_text)

    # Assemble the augmented "user" message that contains the memories block.
    augmented_user = (
        "## Retrieved memories\n"
        f"{memories_text}\n\n"
        "## Current question\n"
        f"{user_text}"
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_messages)
    messages.append({"role": "user", "content": augmented_user})

    return messages, used


def build_naive_rag_messages(
    *,
    system_prompt: str,
    recent_turns: list[dict],
    chunks: list[RankedHit],
    user_message: str,
) -> tuple[list[dict], int]:
    """Build messages for the naive RAG agent."""
    used = 0
    sys_tokens = count_tokens(system_prompt)
    if sys_tokens > config.BUDGET_SYSTEM:
        system_prompt = _truncate_to_tokens(system_prompt, config.BUDGET_SYSTEM)
        sys_tokens = config.BUDGET_SYSTEM
    used += sys_tokens

    # Chunks: keep top-k as listed (already top-k by cosine).
    kept = chunks[: config.CHUNK_TOP_K]
    chunks_text = _format_chunks(kept)
    # Truncate to budget.
    while count_tokens(chunks_text) > config.BUDGET_MEMORIES and kept:
        kept = kept[:-1]
        chunks_text = _format_chunks(kept)
    used += count_tokens(chunks_text)

    chat_tokens = 0
    chat_messages: list[dict] = []
    for turn in reversed(recent_turns):
        block = {"role": turn["role"], "content": turn["text"]}
        t = count_tokens(block["content"]) + 4
        if chat_tokens + t > config.BUDGET_CHAT_HISTORY:
            break
        chat_messages.append(block)
        chat_tokens += t
    chat_messages.reverse()
    used += chat_tokens

    user_text = _truncate_to_tokens(user_message, config.BUDGET_USER_MSG)
    used += count_tokens(user_text)

    augmented_user = (
        "## Relevant document excerpts\n"
        f"{chunks_text}\n\n"
        "## Current question\n"
        f"{user_text}"
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_messages)
    messages.append({"role": "user", "content": augmented_user})

    return messages, used
