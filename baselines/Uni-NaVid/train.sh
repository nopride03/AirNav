#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-uninavid}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

bash ./scripts/uninavid_stage_2.sh
