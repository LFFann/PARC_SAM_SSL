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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/PARC_SAM_SSL_v4_ProtoPrompt_UPSC_RTX3090_24G_echoData}"
MAX_ITERATIONS="${MAX_ITERATIONS:-22000}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/echoData/260513_data_labeled30pct}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-/root/autodl-tmp/sam_vit_b_01ec64.pth}"

python train.py \
  --config "${CONFIG}" \
  --device cuda \
  --max-iterations "${MAX_ITERATIONS}" \
  --data-root "${DATA_ROOT}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}"
