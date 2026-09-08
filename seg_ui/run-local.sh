#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

if [ ! -d node_modules ]; then
  npm install
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r backend/requirements.txt

cleanup() {
  kill "$API_PID" "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python -m uvicorn backend.app:app --host 127.0.0.1 --port 7860 &
API_PID=$!
npm run dev &
UI_PID=$!
wait "$API_PID" "$UI_PID"
