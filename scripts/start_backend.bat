@echo off
REM ---------------------------------------------------------------------------
REM start_backend.bat — Start the Retail AI FastAPI backend (development)
REM ---------------------------------------------------------------------------
echo Starting Retail AI Backend...
cd ..
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
