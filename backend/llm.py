"""Ollama HTTP client. Chat (streaming + non-streaming) and structured JSON.

We deliberately don't use the official `ollama` Python package so the rest of
the codebase depends only on `requests`. The Ollama HTTP API is small enough
that this is fine.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Iterator

import requests

from . import config


log = logging.getLogger(__name__)


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama server can't be reached."""


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self._host = (host or config.OLLAMA_HOST).rstrip("/")
        self._model = model or config.CHAT_MODEL

    # --- low-level ---

    def _post(self, path: str, payload: dict, stream: bool = False, timeout: int = 600):
        url = f"{self._host}{path}"
        try:
            return requests.post(url, json=payload, stream=stream, timeout=timeout)
        except requests.exceptions.ConnectionError as e:
            raise OllamaUnavailable(
                f"Could not reach Ollama at {self._host}. "
                f"Is `ollama serve` running? Underlying error: {e}"
            ) from e

    def _check_health(self) -> None:
        # Cheap health probe; only fail when really unreachable.
        try:
            r = requests.get(f"{self._host}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise OllamaUnavailable(
                f"Ollama not reachable at {self._host}. "
                f"Run `ollama serve` and `ollama pull {self._model}`. "
                f"Underlying error: {e}"
            ) from e

    # --- high-level ---

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        model: str | None = None,
    ) -> str:
        """Call /api/chat. If stream=True and on_token given, stream tokens
        into the callback as they arrive. Always returns the full reply text.
        """
        self._check_health()
        payload = {
            "model": model or self._model,
            "messages": messages,
            "stream": stream,
        }
        if stream:
            return self._chat_stream(payload, on_token)
        return self._chat_blocking(payload)

    def _chat_blocking(self, payload: dict) -> str:
        r = self._post("/api/chat", payload, stream=False, timeout=600)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")

    def _chat_stream(self, payload: dict, on_token: Callable[[str], None] | None) -> str:
        # NDJSON streaming per Ollama spec.
        r = self._post("/api/chat", payload, stream=True, timeout=600)
        r.raise_for_status()
        full: list[str] = []
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                full.append(token)
                if on_token is not None:
                    on_token(token)
            if chunk.get("done"):
                break
        return "".join(full)

    def chat_stream_iter(
        self, messages: list[dict[str, str]], *, model: str | None = None
    ) -> Iterator[str]:
        """Generator that yields tokens as they arrive. Use with SSE."""
        self._check_health()
        payload = {
            "model": model or self._model,
            "messages": messages,
            "stream": True,
        }
        r = self._post("/api/chat", payload, stream=True, timeout=600)
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
            if chunk.get("done"):
                break

    # --- structured output ---

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Call chat and parse the reply as JSON, tolerating code-fenced JSON.

        Used by the consolidator.
        """
        raw = self.chat(messages, stream=False, model=model)
        return _parse_json_lenient(raw)


# --- JSON helpers ---


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json_lenient(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a chat reply that may include prose or fences."""
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    # Fast path: reply is already valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first '{' and the matching '}'.
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in reply: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break
    # Last-ditch: log the raw text and raise.
    log.warning("Failed to parse JSON from LLM reply: %r", text[:500])
    raise ValueError(f"Failed to parse JSON from LLM reply: {text[:200]!r}")


# Module-level singleton; agent modules import this.
client = OllamaClient()
