@echo off
rem Chess Diagram -> PGN launcher (Windows)
cd /d "%~dp0"
if not exist .venv (
  echo First run: creating virtual environment and installing dependencies...
  python -m venv .venv || (echo Python 3.9+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
)
.venv\Scripts\python app\app.py %*
pause
