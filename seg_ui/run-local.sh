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

# Windows ComfyUI（RTX 4070 Ti SUPER，经 Tailscale）。可用环境变量覆盖。
export COMFYUI_URL="${COMFYUI_URL:-https://win-hm9ig3vhnaa.tailfff622.ts.net}"
export COMFYUI_ENGINE="${COMFYUI_ENGINE:-sd15}"
export COMFYUI_CHECKPOINT="${COMFYUI_CHECKPOINT:-sd-v1-5-inpainting.safetensors}"
export COMFYUI_LAMA_MODEL="${COMFYUI_LAMA_MODEL:-big-lama.pt}"
export COMFYUI_WORK_SIZE="${COMFYUI_WORK_SIZE:-768}"

cleanup() {
  kill "$API_PID" "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(
  # Load private credentials only into the backend, not npm or the frontend.
  if [ -f "$APP_DIR/config/qwen.env" ]; then
    set -a
    . "$APP_DIR/config/qwen.env"
    set +a
  fi
  exec python -m uvicorn backend.app:app --host 127.0.0.1 --port 7860
) &
API_PID=$!
npm run dev &
UI_PID=$!
wait "$API_PID" "$UI_PID"
