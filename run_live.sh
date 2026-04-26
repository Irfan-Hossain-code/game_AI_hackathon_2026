#!/usr/bin/env bash
# One command: ensure Ollama is up, then stream the agent (pass flags to live_agent.py).
# Examples:
#   ./run_live.sh "What is 2+2?"
#   ./run_live.sh "Hello" --kokoro
#   ./run_live.sh "Hello" --f5
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

ollama_up() {
  curl -sS -m 2 "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1
}

if ! ollama_up; then
  echo "[run_live] Ollama not responding at ${OLLAMA_HOST} — starting \`ollama serve\` in the background…"
  ollama serve >>/tmp/ollama-serve-run_live.log 2>&1 &
  disown 2>/dev/null || true
  for _ in $(seq 1 120); do
    if ollama_up; then
      echo "[run_live] Ollama is up."
      break
    fi
    sleep 0.25
  done
  if ! ollama_up; then
    echo "[run_live] Could not reach Ollama. Start it yourself: ollama serve"
    echo "[run_live] Log: /tmp/ollama-serve-run_live.log"
    exit 1
  fi
fi

exec python3 live_agent.py "$@"
