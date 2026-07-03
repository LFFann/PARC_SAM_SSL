#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
if ! [[ "${OMP_NUM_THREADS:-4}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=4
fi
if ! [[ "${MKL_NUM_THREADS:-4}" =~ ^[0-9]+$ ]]; then
  export MKL_NUM_THREADS=4
fi

CONFIG="${CONFIG:-configs/parc_sam_ssl_3090_24g_echo.yaml}"
RUN_DIR="${RUN_DIR:-outputs/PARC_SAM_SSL_v4_ProtoPrompt_UPSC_RTX3090_24G_echoData}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoints/best.pt}"
SPLIT="${SPLIT:-test}"
SAVE_DIR="${SAVE_DIR:-${RUN_DIR}/prediction_${SPLIT}}"

python evaluate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --device cuda \
  --save-dir "${SAVE_DIR}"
