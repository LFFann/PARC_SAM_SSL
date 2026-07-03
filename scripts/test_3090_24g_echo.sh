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

CONFIG="${CONFIG:-configs/parc_sam_ssl_3090_24g_echo.yaml}"
RUN_DIR="${RUN_DIR:-outputs/PARC_SAM_SSL_v4_ProtoPrompt_UPSC_RTX3090_24G_echoData}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoints/best.pt}"
SPLIT="${SPLIT:-test}"
SAVE_DIR="${SAVE_DIR:-${RUN_DIR}/prediction_${SPLIT}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  if [[ "${CHECKPOINT}" == "${RUN_DIR}/checkpoints/best.pt" && -f "${RUN_DIR}/checkpoints/final.pt" ]]; then
    CHECKPOINT="${RUN_DIR}/checkpoints/final.pt"
    echo "[PARC-SAM-SSL] best.pt not found; using final.pt: ${CHECKPOINT}" >&2
  else
    echo "[PARC-SAM-SSL] checkpoint not found: ${CHECKPOINT}" >&2
    echo "[PARC-SAM-SSL] train first with: bash scripts/train_3090_24g_echo.sh" >&2
    echo "[PARC-SAM-SSL] expected checkpoints under: ${RUN_DIR}/checkpoints" >&2
    if [[ -d "${RUN_DIR}/checkpoints" ]]; then
      find "${RUN_DIR}/checkpoints" -maxdepth 1 -type f -name "*.pt" -print >&2
    fi
    exit 2
  fi
fi

python evaluate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --device cuda \
  --save-dir "${SAVE_DIR}"
