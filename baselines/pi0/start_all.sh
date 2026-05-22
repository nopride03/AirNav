#!/usr/bin/env bash
# Simplified launcher that bypasses health checks.
# Starts backends and gateway directly without waiting for curl checks.

set -euo pipefail

GPU_LIST="${GPU_LIST:-0,1}"
BASE_PORT="${BASE_PORT:-8050}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"
HOST="${HOST:-0.0.0.0}"
TIMEOUT="${TIMEOUT:-120}"

OPENPI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ENV="${OPENPI_ENV:-${OPENPI_DIR}/.venv}"
PY="${PYTHON:-python}"
if [[ -x "${OPENPI_ENV}/bin/python" ]]; then
    PY="${OPENPI_ENV}/bin/python"
fi

if [[ -d "${OPENPI_ENV}" ]]; then
    export PATH="${OPENPI_ENV}/bin:${PATH}"
    export PKG_CONFIG_PATH="${OPENPI_ENV}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
    export LD_LIBRARY_PATH="${OPENPI_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export OPENPI_CONFIG="${OPENPI_CONFIG:-pi0_vln_lora}"
if [[ -n "${OPENPI_CHECKPOINT:-}" ]]; then
    export OPENPI_CHECKPOINT
fi

cd "${OPENPI_DIR}"

IFS=',' read -ra GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "[start_all.sh] No GPUs configured. Set GPU_LIST." >&2
    exit 1
fi

pids=()

cleanup() {
    if [ "${#pids[@]}" -gt 0 ]; then
        echo "[start_all.sh] Stopping processes: ${pids[*]}"
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

mkdir -p logs

# Start backends
backend_urls=()
for i in "${!GPUS[@]}"; do
    gpu="$(echo "${GPUS[$i]}" | xargs)"
    port=$((BASE_PORT + i))
    log="logs/backend_gpu${gpu}_port${port}.log"
    echo "[start_all.sh] Starting backend on GPU ${gpu} port ${port}  (log: ${log})"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PY}" api.py --host "${HOST}" --port "${port}" \
        > "${log}" 2>&1 &
    pids+=("$!")
    backend_urls+=("http://127.0.0.1:${port}")
done

# Wait a bit for backends to initialize (no health check, just fixed delay)
echo "[start_all.sh] Waiting 120 seconds for backends to load..."
sleep 120

# Start gateway
BACKEND_URLS="$(IFS=,; echo "${backend_urls[*]}")"
export BACKEND_URLS

echo "[start_all.sh] Starting gateway on ${HOST}:${GATEWAY_PORT}"
"${PY}" gateway.py --host "${HOST}" --port "${GATEWAY_PORT}" --timeout "${TIMEOUT}" \
    > "logs/gateway_port${GATEWAY_PORT}.log" 2>&1 &
pids+=("$!")

echo "[start_all.sh] Gateway:  http://127.0.0.1:${GATEWAY_PORT}"
echo "[start_all.sh] Backends: ${BACKEND_URLS}"
echo "[start_all.sh] Press Ctrl+C to stop everything."
echo "[start_all.sh] Check logs/ for backend/gateway output."

wait "${pids[@]}"
