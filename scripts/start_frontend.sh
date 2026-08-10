#!/bin/bash
# ---------------------------------------------------------------------------
# start_frontend.sh — Start the Retail AI Streamlit frontend (development)
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Starting Retail AI Frontend..."
uv run streamlit run frontend/Home.py --server.port 8501
