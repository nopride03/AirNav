#! /bin/bash
# Launcher for pi0 fine-tuning on the VLN dataset (the same JSON consumed by
# the OpenFly / OpenVLA / NaVid / Uni-NaVid baselines).
#
# Usage
# -----
#   bash train_pi0.sh smoke                   # 50-step smoke test on the 1024-sample subset
#   bash train_pi0.sh lora [extra tyro args]  # LoRA fine-tune pi0_base on the full 618k-sample dataset
#   bash train_pi0.sh stats <config>          # (re)compute norm stats for <config>
#
# Training length is expressed in **epochs** (of the LeRobot / JSON dataset).
# The launcher looks up the dataset size and the effective batch size from the
# TrainConfig, and converts to the number of optimizer steps that
# ``scripts/train.py`` consumes internally. Override via env vars:
#
#   NUM_EPOCHS=2  bash train_pi0.sh lora            # train for 2 epochs (default: 1)
#   BATCH_SIZE=16 bash train_pi0.sh lora            # override batch size; steps recomputed
#   EXP_NAME=my_run bash train_pi0.sh lora          # custom experiment id
#   SAVE_PER_EPOCH=4 bash train_pi0.sh lora         # save 4 ckpts per epoch (default: 2)
#
# Additional tyro flags pass through after the mode, e.g.
#   bash train_pi0.sh lora --learning_rate 5e-5
#
# Key assumptions
# ---------------
# *   A Python environment has been populated via ``uv sync``.
# *   ``OPENPI_VLN_DATA_DIR`` points to the directory containing
#     ``train_converted.json`` and ``train_1024_converted.json``.
# *   ``OPENPI_PI0_BASE_CHECKPOINT`` points to the pi0_base Orbax checkpoint,
#     or the default public ``gs://openpi-assets/checkpoints/pi0_base/params``
#     location is accessible.

set -euo pipefail

MODE="${1:-smoke}"
shift || true

OPENPI_DIR="${OPENPI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
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
export OPENPI_VLN_DATA_DIR="${OPENPI_VLN_DATA_DIR:-data/vln}"
export OPENPI_PI0_BASE_CHECKPOINT="${OPENPI_PI0_BASE_CHECKPOINT:-gs://openpi-assets/checkpoints/pi0_base/params}"
# Make sure an inherited ``JAX_PLATFORMS=cpu`` from a previous norm-stats run
# doesn't silently force pi0 onto the CPU (we want CUDA here).
unset JAX_PLATFORMS
# Pi0 uses jax[cuda12]. Prevent JAX from preallocating the whole VRAM so
# PyTorch data loader workers can also use the GPUs for image decode etc.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
# Silence the lerobot / huggingface download warnings about GIT-LFS.
export GIT_LFS_SKIP_SMUDGE=1
# Don't preallocate on workers either.
export TOKENIZERS_PARALLELISM=false

cd "${OPENPI_DIR}"

_ensure_stats() {
    local config="$1"
    local stats_path="${OPENPI_DIR}/assets/${config}/vln_norm/norm_stats.json"
    if [[ ! -f "${stats_path}" ]]; then
        echo "[train_pi0] computing norm stats for ${config} (missing ${stats_path})"
        "${PY}" scripts/compute_norm_stats.py --config-name "${config}" "${@:2}"
    else
        echo "[train_pi0] reusing norm stats at ${stats_path}"
    fi
}

# Ask the TrainConfig / JSON dataset how many optimizer steps correspond to
# ``$NUM_EPOCHS`` epochs and ``$BATCH_SIZE`` per global batch. Echoes four
# integers separated by spaces: ``<num_steps> <save_interval> <log_interval> <keep_period>``.
#
# Outputs are derived so that:
#   * at least 1 checkpoint per epoch (``SAVE_PER_EPOCH=2`` by default),
#   * at least ~200 loss log lines across the whole run,
#   * permanent checkpoints kept every epoch boundary.
_compute_schedule() {
    local config="$1"
    local num_epochs="$2"
    local batch_size="$3"
    local save_per_epoch="${4:-2}"
    "${PY}" - "${config}" "${num_epochs}" "${batch_size}" "${save_per_epoch}" <<'PYEOF'
import json
import math
import sys
from pathlib import Path

import openpi.training.config as _config

config_name, num_epochs, batch_size, save_per_epoch = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
)
cfg = _config.get_config(config_name)

repo_id = cfg.data.repo_id
if repo_id is None or not repo_id.startswith("vln_json:"):
    raise SystemExit(f"_compute_schedule only supports vln_json configs, got repo_id={repo_id!r}")

json_path = Path(repo_id[len("vln_json:") :])
with json_path.open("r") as f:
    num_samples = len(json.load(f))

steps_per_epoch = max(1, math.ceil(num_samples / max(1, batch_size)))
num_train_steps = steps_per_epoch * num_epochs
save_interval = max(100, steps_per_epoch // max(1, save_per_epoch))
log_interval = max(10, num_train_steps // 500)
keep_period = max(save_interval, steps_per_epoch)

sys.stderr.write(
    f"[schedule] dataset={json_path.name} samples={num_samples} batch={batch_size} "
    f"epochs={num_epochs} steps_per_epoch={steps_per_epoch} total_steps={num_train_steps} "
    f"save_every={save_interval} log_every={log_interval} keep_every={keep_period}\n"
)
print(num_train_steps, save_interval, log_interval, keep_period)
PYEOF
}

case "${MODE}" in
    smoke)
        CONFIG="pi0_vln_smoke"
        EXP_NAME_DEFAULT="smoke-$(date +%m%d-%H%M)"
        BATCH_SIZE="${BATCH_SIZE:-8}"
        NUM_EPOCHS="${NUM_EPOCHS:-1}"
        _ensure_stats "${CONFIG}"
        # Schedule is computed from (dataset_size / batch) * epochs.
        read -r NUM_STEPS SAVE_EVERY LOG_EVERY KEEP_EVERY < <(_compute_schedule "${CONFIG}" "${NUM_EPOCHS}" "${BATCH_SIZE}" 4)
        echo "[train_pi0] smoke test: ${CONFIG} exp=${EXP_NAME:-$EXP_NAME_DEFAULT} epochs=${NUM_EPOCHS} steps=${NUM_STEPS}"
        "${PY}" scripts/train.py "${CONFIG}" \
            --exp_name "${EXP_NAME:-$EXP_NAME_DEFAULT}" \
            --batch_size "${BATCH_SIZE}" \
            --num_train_steps "${NUM_STEPS}" \
            --save_interval "${SAVE_EVERY}" \
            --log_interval "${LOG_EVERY}" \
            --keep_period "${KEEP_EVERY}" \
            --overwrite \
            --no-wandb_enabled \
            "$@"
        ;;

    lora)
        CONFIG="pi0_vln_lora"
        EXP_NAME_DEFAULT="pi0-vln-lora-$(date +%m%d-%H%M)"
        BATCH_SIZE="${BATCH_SIZE:-8}"
        NUM_EPOCHS="${NUM_EPOCHS:-1}"
        SAVE_PER_EPOCH="${SAVE_PER_EPOCH:-2}"
        # ``FSDP_DEVICES`` controls how many devices the model is sharded across.
        # If the JAX device count is <= FSDP_DEVICES the whole model is sharded
        # and there is no data parallelism; otherwise data parallelism runs
        # across ``num_devices / FSDP_DEVICES`` groups.
        FSDP_DEVICES="${FSDP_DEVICES:-8}"
        # For the full dataset we sample max 50k frames for stats (fast enough
        # while still covering the full state distribution). Users can override
        # by deleting ``assets/pi0_vln_lora/vln_norm``.
        _ensure_stats "${CONFIG}" --max-frames 50000
        read -r NUM_STEPS SAVE_EVERY LOG_EVERY KEEP_EVERY < <(_compute_schedule "${CONFIG}" "${NUM_EPOCHS}" "${BATCH_SIZE}" "${SAVE_PER_EPOCH}")
        echo "[train_pi0] LoRA fine-tune: ${CONFIG} exp=${EXP_NAME:-$EXP_NAME_DEFAULT} epochs=${NUM_EPOCHS} steps=${NUM_STEPS} fsdp=${FSDP_DEVICES} batch=${BATCH_SIZE}"
        "${PY}" scripts/train.py "${CONFIG}" \
            --exp_name "${EXP_NAME:-$EXP_NAME_DEFAULT}" \
            --fsdp_devices "${FSDP_DEVICES}" \
            --batch_size "${BATCH_SIZE}" \
            --num_train_steps "${NUM_STEPS}" \
            --save_interval "${SAVE_EVERY}" \
            --log_interval "${LOG_EVERY}" \
            --keep_period "${KEEP_EVERY}" \
            "$@"
        ;;

    stats)
        CONFIG="${1:-pi0_vln_smoke}"
        shift || true
        echo "[train_pi0] computing stats for ${CONFIG}"
        rm -f "${OPENPI_DIR}/assets/${CONFIG}/vln_norm/norm_stats.json"
        "${PY}" scripts/compute_norm_stats.py --config-name "${CONFIG}" "$@"
        ;;

    *)
        echo "Unknown mode '${MODE}'. Expected one of: smoke | lora | stats"
        exit 1
        ;;
esac
