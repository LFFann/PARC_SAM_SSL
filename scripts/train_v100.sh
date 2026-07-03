#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python train.py \
  --config configs/parc_sam_ssl_3class.yaml \
  --device cuda \
  --max-iterations "${MAX_ITERATIONS:-10000}" \
  --output-dir "${OUTPUT_DIR:-outputs/PARC_SAM_SSL_3Class}"

