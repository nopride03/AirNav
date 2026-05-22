#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0}}"
BASE_PORT="${BASE_PORT:-8050}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"
HOST="${HOST:-0.0.0.0}"
TIMEOUT="${TIMEOUT:-120}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/uninavid}"

IFS=',' read -ra GPUS <<< "$GPU_LIST"

if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "No GPUs configured. Set GPU_LIST or CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

pids=()

cleanup() {
  if [ "${#pids[@]}" -gt 0 ]; then
    echo "Stopping processes: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

for i in "${!GPUS[@]}"; do
  gpu="$(echo "${GPUS[$i]}" | xargs)"
  port=$((BASE_PORT + i))
  echo "Starting backend on GPU ${gpu} port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" python api.py \
    --host "${HOST}" \
    --port "${port}" \
    --model-path "${MODEL_PATH}" &
  pids+=("$!")
done

backend_urls=()
for i in "${!GPUS[@]}"; do
  port=$((BASE_PORT + i))
  backend_urls+=("[REDACTED_URL]}")
done

BACKEND_URLS="$(IFS=,; echo "${backend_urls[*]}")"
export BACKEND_URLS

echo "Starting gateway on port ${GATEWAY_PORT}"
python gateway.py --host "${HOST}" --port "${GATEWAY_PORT}" --timeout "${TIMEOUT}" &
pids+=("$!")

echo "Gateway ready at [REDACTED_URL]}"
echo "Backends: ${BACKEND_URLS}"
echo "Model: ${MODEL_PATH}"

wait "${pids[@]}"