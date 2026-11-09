#!/usr/bin/env bash
# Chess Diagram -> PGN launcher (macOS / Linux)
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "First run: creating virtual environment and installing dependencies..."
  $PY -m venv .venv
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python app/app.py "$@"
