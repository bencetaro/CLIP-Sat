#!/usr/bin/env bash
set -euo pipefail

cpp_path="${LLM_CPP_MODEL_PATH:-/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
cpp_url="${LLM_CPP_MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf}"

mkdir -p "$(dirname "$cpp_path")"
mkdir -p /app/media/wandb_artifacts

if [ ! -f "$cpp_path" ]; then
    echo "[startup] Downloading llama.cpp GGUF to: $cpp_path"
    if [ -n "${HF_TOKEN:-}" ] && [ "${HF_TOKEN}" != "?" ]; then
        curl -fL -H "Authorization: Bearer ${HF_TOKEN}" -o "$cpp_path" "$cpp_url"
    else
        curl -fL -o "$cpp_path" "$cpp_url"
    fi
else
    echo "[startup] GGUF already present at: $cpp_path"
fi

echo "[startup] Models ready."
exec uvicorn src.inference.api_service:app --host 0.0.0.0 --port 8000
