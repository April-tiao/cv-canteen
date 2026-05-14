#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/head_only.yaml}"
METHOD="${2:-head_only}"
GPU="${GPU:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${STAMP}_${METHOD}"
LOG_DIR="logs/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/train.log"

mkdir -p "${LOG_DIR}"

{
  echo "run_name=${RUN_NAME}"
  echo "config=${CONFIG}"
  echo "gpu=${GPU}"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "dataset_counts:"
  find dataset/train/food -type f | wc -l | xargs echo "train food"
  find dataset/train/chart -type f | wc -l | xargs echo "train chart"
  find dataset/train/other -type f | wc -l | xargs echo "train other"
  find dataset/val/food -type f | wc -l | xargs echo "val food"
  find dataset/val/chart -type f | wc -l | xargs echo "val chart"
  find dataset/val/other -type f | wc -l | xargs echo "val other"
} | tee "${LOG_DIR}/run_info.txt"

nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU} python -u train.py --config '${CONFIG}' --device cuda --run-name '${RUN_NAME}'" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

PID="$!"
echo "${PID}" | tee "${LOG_DIR}/pid.txt"
echo "started pid=${PID}"
echo "log=${LOG_FILE}"
echo "checkpoint_dir=checkpoints/${RUN_NAME}"
echo "tail -f ${LOG_FILE}"
