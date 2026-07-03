#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

sanitize_threads() {
  local name="$1"
  local default="$2"
  local value="${!name:-$default}"
  value="${value//$'\r'/}"
  value="${value//$'\n'/}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -lt 1 ]]; then
    value="${default}"
  fi
  export "${name}=${value}"
}

sanitize_threads OMP_NUM_THREADS 4
sanitize_threads MKL_NUM_THREADS 4

CONFIG="${CONFIG:-configs/parc_sam_ssl_v100_32g_echo.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/PARC_SAM_SSL_v4_ProtoPrompt_UPSC_V100_32G_echoData}"
MAX_ITERATIONS="${MAX_ITERATIONS:-18000}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/echoData/260703_data_labeled30pct}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-/root/autodl-tmp/sam_vit_b_01ec64.pth}"

python train.py \
  --config "${CONFIG}" \
  --device cuda \
  --max-iterations "${MAX_ITERATIONS}" \
  --data-root "${DATA_ROOT}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}"
