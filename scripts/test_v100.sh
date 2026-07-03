#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python evaluate.py \
  --config configs/parc_sam_ssl_3class.yaml \
  --checkpoint "${CHECKPOINT:-outputs/PARC_SAM_SSL_3Class/checkpoints/best.pt}" \
  --split test \
  --device cuda \
  --save-dir "${SAVE_DIR:-outputs/PARC_SAM_SSL_3Class/prediction_test}"

