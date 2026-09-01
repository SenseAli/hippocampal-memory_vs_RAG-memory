#!/usr/bin/env bash
# One-command launcher for the hippocampal-agent demo.
# Prereqs: Ollama installed and `ollama serve` running, and `llama3` pulled.
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 "$@"
