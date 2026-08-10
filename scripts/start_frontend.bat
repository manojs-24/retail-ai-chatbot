@echo off
REM ---------------------------------------------------------------------------
REM start_frontend.bat — Start the Retail AI Streamlit frontend (development)
REM ---------------------------------------------------------------------------
echo Starting Retail AI Frontend...
cd ..
uv run streamlit run frontend/Home.py --server.port 8501
