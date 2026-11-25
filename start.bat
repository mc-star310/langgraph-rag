@echo off
REM ============================================================
REM  langgraph-rag one-click starter (Windows)
REM  Run on your machine, NOT in the sandbox.
REM  1) install Ollama from https://ollama.com
REM  2) run:  ollama pull bge-m3
REM  3) double-click this file
REM ============================================================
where ollama >nul 2>nul || (echo [ERROR] Ollama not found. Install from https://ollama.com && pause && exit /b)

start "" ollama serve
timeout /t 5 >nul

if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install -q -r requirements.txt

start "" http://localhost:8000
uvicorn app:app --host 0.0.0.0 --port 8000
