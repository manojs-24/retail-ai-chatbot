#!/bin/bash
# ---------------------------------------------------------------------------
# start_backend.sh — Start the Retail AI FastAPI backend (development)
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Starting Retail AI Backend..."
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
